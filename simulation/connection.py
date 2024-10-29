import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from simulation.settings import PORT, TIMEOUT, HOST


class ClientConnection:
    def __init__(self, town):
        self.town = town
        self.client = None
        self.world = None

    def setup(self):
        """
        Setup the client connection and the world.
        """
        try:

            self.client = carla.Client(HOST, PORT)
            self.client.set_timeout(TIMEOUT)
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)
            return self.client, self.world

        except Exception as e:
            print(f'Failed to make a connection with the server: {e}')
            self.error()

    def error(self):
        """
        Print client and server versions.
        """

        print(f"\nClient version: {self.client.get_client_version()}")
        print(f"Server version: {self.client.get_server_version()}\n")

        if self.client.get_client_version != self.client.get_server_version:
            print(
                "There is a Client and Server version mismatch! Please install or download the right versions.")
