#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
import urllib
import os
import sys


def get_soup(url):
    return BeautifulSoup(requests.get(url).text, 'lxml')


def main(args):
    query = args[0]

    url = "https://www.google.co.jp/search?q=" + query + \
        "&source=lnms&tbm=isch&sa=X&ved=0ahUKEwi0lKDl3pTUAhVFzLwKHateDc8Q_AUICigB&biw=885&bih=826"

    soup = get_soup(url)
    images = [a['src'] for a in soup.find_all("img", {"src": re.compile("image")})]

    for idx, img in enumerate(images):
        raw_img = urllib.request.urlopen(img).read()
        with open("images/" + query + "_" + str(idx) + '.jpg', 'wb') as f:
            f.write(raw_img)


if __name__ == '__main__':
    main(sys.argv[1:])
