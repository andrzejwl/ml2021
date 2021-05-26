import requests
import os
from urllib.parse import urlparse


def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def download(response, pathname, year):
    if not os.path.isdir(pathname):
        os.makedirs(pathname)

    filename = os.path.join(pathname, f"{str(year)}_" + response.url.split("/")[-1])

    with open(filename, "wb") as f:
        f.write(response.content)
    print('Downloaded: ' + response.url)


# 'https://graf.kzp.pl/kat/zn/1952/zn0583.jpg'
def scrap():
    # 1952/583
    # 2010/4315
    # 2018/4822
    year = 2010
    stampNum = 4315
    yearUpperBound = 2019

    reset_counter = 0

    url = constructUrl(year, stampNum)
    for i in range(year, yearUpperBound):
        while is_valid(url):
            if reset_counter == 3:
                stampNum -= reset_counter
                reset_counter = 0
                break
            url = constructUrl(i, stampNum)
            print(url)
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 404:
                print('NOT FOUND 404')
                reset_counter +=1
            else:
                reset_counter = 0
                download(response, 'images', i)
            stampNum = stampNum + 1


def constructUrl(year, number):
    baseUrl = 'https://graf.kzp.pl/kat/zn/'
    if number < 1000:
        url = baseUrl + str(year) + '/zn0' + str(number) + '.jpg'
    else:
        url = baseUrl + str(year) + '/zn' + str(number) + '.jpg'
    return url


if __name__ == "__main__":
    scrap()
