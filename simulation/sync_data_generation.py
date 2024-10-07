#!/usr/bin/env python

import glob
import os
import sys
from scipy.spatial.transform import Rotation as R
import math
from carla import ColorConverter

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def setCameraParams(camera):
    sizeX = 1280
    sizeY = 720
    fov = 110

    camera.set_attribute('image_size_x', str(sizeX))
    camera.set_attribute('image_size_y', str(sizeY))
    camera.set_attribute('fov', str(fov))
    if camera.has_attribute('motion_blur_max_distortion'):
        camera.set_attribute('motion_blur_max_distortion', '0')

    return camera


def carla_transform_to_mat(carla_transform):
    """
    Convert a carla transform from a left-handed X-forward system (unreal)
    to a right-handed Z-forward camera pose

    :param carla_transform: the carla transform
    :return: a numpy.array with 4x4 pose matrix
    """
    camToWorld = np.matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    worldToCam = np.matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

    # Identity pose
    mat = np.eye(4)

    # Position
    mat[:3, 3] = worldToCam @ np.array([
        carla_transform.location.x,
        -carla_transform.location.y,
        carla_transform.location.z])

    # Rotation
    roll = carla_transform.rotation.roll
    pitch = -carla_transform.rotation.pitch
    yaw = -carla_transform.rotation.yaw
    worldCS = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()

    mat[:3, :3] = worldToCam @ worldCS @ camToWorld

    return mat

def append_camera_pose(frame_id, transform):
    # time,x,y,z,lat,lon,altitude,pitch,yaw,roll
    T = carla_transform_to_mat(transform)

    rot = R.from_matrix(T[:3, :3]).as_quat()
    trans = T[:3, 3]
    with open('data/gt.tum','a+') as f:
        f.write('%s %s %s %s %s %s %s %s\n' % (frame_id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]))

def main(display_enabled=False):
    # Parameters
    record_fps = 6  # Define the record FPS variable
    T_vehicle_camera = carla.Transform(carla.Location(x=1, y=0.0, z=3), carla.Rotation(pitch=10, roll=0, yaw=0))

    actor_list = []
    if display_enabled:
        pygame.init()
        display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Iterate over all towns
    towns = client.get_available_maps()
    print(f"Available towns: {towns}")
    for town in towns:
        print(f"Loading town: {town}")
        client.load_world(town)
        world = client.get_world()

        # Iterate over all weather presets
        weather_presets = [(name, param) for name, param in carla.WeatherParameters.__dict__.items() if isinstance(param, carla.WeatherParameters)]
        for weather_name, weather_param in weather_presets:
            # Check if the path for this town and weather combination already exists
            town_name = os.path.basename(town)  # Extracts 'Default' from the path
            base_path = f'data/rgb/{town_name}/{weather_name}'
            print(f"Checking path: {base_path}")
            if os.path.exists(base_path):
                print(f"Skipping {town_name} with weather {weather_name} as data already exists.")
                continue

            print(f"Setting weather: {weather_name}")
            world.set_weather(weather_param)

            try:
                m = world.get_map()
                spawn_points = m.get_spawn_points()
                blueprint_library = world.get_blueprint_library()

                # Spawn the main vehicle
                vehicle_bp = blueprint_library.find('vehicle.audi.tt')
                vehicle = None
                for attempt in range(10):  # Try up to 10 different spawn points
                    spawn_point = random.choice(spawn_points)
                    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                    if vehicle:
                        actor_list.append(vehicle)
                        vehicle.set_simulate_physics(False)
                        print(f"Spawned main vehicle at attempt {attempt + 1}.")
                        break
                if not vehicle:
                    print("Failed to spawn main vehicle after 10 attempts.")
                    continue

                # Spawn additional vehicles
                num_vehicles = 5
                for i in range(num_vehicles):
                    spawn_point = random.choice(spawn_points)
                    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
                    additional_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                    if additional_vehicle:
                        actor_list.append(additional_vehicle)

                # Spawn pedestrians
                num_pedestrians = 10
                walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
                for i in range(num_pedestrians):
                    spawn_point = random.choice(spawn_points)
                    pedestrian = world.try_spawn_actor(walker_bp, spawn_point)
                    if pedestrian:
                        actor_list.append(pedestrian)

                # Attach cameras to the main vehicle
                camera_rgb = world.spawn_actor(
                    setCameraParams(blueprint_library.find('sensor.camera.rgb')),
                    T_vehicle_camera,
                    attach_to=vehicle)
                actor_list.append(camera_rgb)

                camera_depth = world.spawn_actor(
                    setCameraParams(blueprint_library.find('sensor.camera.depth')),
                    T_vehicle_camera,
                    attach_to=vehicle)
                actor_list.append(camera_depth)

                camera_semseg = world.spawn_actor(
                    setCameraParams(blueprint_library.find('sensor.camera.semantic_segmentation')),
                    T_vehicle_camera,
                    attach_to=vehicle)
                actor_list.append(camera_semseg)

                tick_counter = 0
                counter = 0
                max_images = 10
                batch_size = 10
                image_batches = {'rgb': [], 'depth': [], 'semseg': []}
                current_location = vehicle.get_location()
                waypoint = m.get_waypoint(current_location)

                # Create a synchronous mode context.
                with CarlaSyncMode(world, camera_rgb, camera_depth, camera_semseg, fps=30) as sync_mode:
                    while counter < max_images:
                        clock.tick()
                        # Advance the simulation and wait for the data.
                        snapshot, image_rgb, image_depth, image_semseg = sync_mode.tick(timeout=2.0)

                        # Choose the next waypoint and update the car location.
                        next_waypoints = waypoint.next(1.5)
                        if not next_waypoints:
                            print("No more waypoints available, breaking loop.")
                            break
                        waypoint = random.choice(next_waypoints)
                        vehicle.set_transform(waypoint.transform)

                        fps = round(1.0 / snapshot.timestamp.delta_seconds)

                        if tick_counter % (30 // record_fps) == 0:
                            counter += 1
                            # Stack images in memory
                            image_batches['rgb'].append(image_rgb)
                            image_batches['depth'].append(image_depth)
                            image_batches['semseg'].append(image_semseg)

                            # Save data in batches
                            if len(image_batches['rgb']) >= batch_size:
                                print(f"Saving batch of images, batch size: {batch_size}.")
                                for image_type, images in image_batches.items():
                                    town_name = os.path.basename(town)  # Extracts 'Default' from the path
                                    base_path = f'data/{image_type}/{town_name}/{weather_name}'
                                    os.makedirs(base_path, exist_ok=True)
                                    for i, image in enumerate(images):
                                        index = counter - batch_size + i
                                        if image_type == 'semseg':
                                            image.save_to_disk(f'{base_path}/{str(index).zfill(3)}', ColorConverter.CityScapesPalette)
                                        elif image_type == 'depth':
                                            image.save_to_disk(f'{base_path}/{str(index).zfill(3)}')
                                        else:
                                            image.save_to_disk(f'{base_path}/{str(index).zfill(3)}')
                                        if image_type == 'rgb':
                                            append_camera_pose(index, camera_rgb.get_transform())
                                    image_batches = {'rgb': [], 'depth': [], 'semseg': []}

                        if display_enabled:
                            # Draw the display.
                            draw_image(display, image_rgb)
                            draw_image(display, image_depth, blend=True)
                            display.blit(
                                font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                                (8, 10))
                            display.blit(
                                font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                                (8, 28))
                            pygame.display.flip()

                        tick_counter += 1

            finally:
                print('Destroying actors.')
                for actor in actor_list:
                    actor.destroy()
                actor_list.clear()

    if display_enabled:
        pygame.quit()
    print('Done.')


if __name__ == '__main__':
    try:
        main(display_enabled=False)  # Set to True to enable display, False to disable
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
