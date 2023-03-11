import requests
import json
import shutil
import os
from os import listdir
from os.path import isfile, join

# HTMS System API Integration
# Student ID: 19045559
# Team Project Assignment Semester B

def download_image(image_url, file_path):
    # Open the url image, set stream to True, this will return the stream content.
    resp = requests.get(image_url, stream=True)
    # Open a local file with wb ( write binary ) permission.
    local_file = open(file_path, 'wb')
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    resp.raw.decode_content = True
    # Copy the response stream raw data to local image file.
    shutil.copyfileobj(resp.raw, local_file)
    # Remove the image url response object.
    del resp


class Profile:
    def __init__(self, data) -> None:
        self.user_id = data['user']['id']
        self.image_name = data['image']['name']
        self.image_url = data['image']['url']
        self.first_name = data['name']['first']
        self.last_name = data['name']['last']
        self.full_name = data['name']['full']

    def get_image_disk_name(self):
        ext = self.get_image_extension()
        return f"{self.full_name}{ext}"

    def get_image_extension(self):
        filename, file_extension = os.path.splitext(self.image_name)

        return file_extension


class ImageDatabase:

    def __init__(self, images_dir, profiles) -> None:
        self.profiles = profiles
        self.images_dir = images_dir

    def sync(self):
        # Download newer files
        whitelist = self.download_images()

        # Clear files that are not in the profiles
        self.clear_disk_images(whitelist)

    def download_images(self):
        whitelist = []
        for profile in self.profiles:
            filename = profile.get_image_disk_name()
            filepath = self.make_disk_path(filename)
            if not os.path.exists(filepath):
                download_image(profile.image_url, filepath)
            whitelist.append(filename)

        return whitelist

    def clear_disk_images(self, whitelist):
        files = self.get_disk_files()
        for file in files:
            if file not in whitelist:
                os.remove(self.make_disk_path(file))

    def make_disk_path(self, filename):
        return f"{self.images_dir}/{filename}"

    def get_disk_files(self):
        return [f for f in listdir(self.images_dir) if isfile(join(self.images_dir, f))]


class API:
    def __init__(self, host) -> None:
        self.host = host
        self.profiles = {}

    def get_profiles(self):
        url = self.url('users/faces/')
        r = requests.get(url, headers=self.__get_headers())
        profiles = r.json()['data']
        self.profiles = []
        for data in profiles:
            profile = Profile(data)
            self.profiles.append(profile)

        return self.profiles

    def sign(self, user_id, site_id):
        url = self.url('attendance/sign')
        data = {
            'user_id':  user_id,
            'site_id': site_id
        }
        r = requests.post(url, data=data, headers=self.__get_headers())
        return r.json()

    def __get_headers(self):
        return {
            'Accept': 'application/json',
            'Content-type': 'application/json'
        }

    def url(self, path):
        return f"{self.host}/api/{path}"
