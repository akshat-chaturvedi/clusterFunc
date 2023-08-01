import os
import hashlib
from urllib.request import urlopen, Request
import logging
from emailer import sendMail

def read_last_hash(fileName: str):
    with open(fileName, 'r', encoding='utf-8') as file:
        return file.read().strip()

def save_hash(fileName: str, new_hash: str):
    with open(fileName, 'w', encoding='utf-8') as file:
        file.write(new_hash)

def webChecker(url: str, fileName: str):
    url = Request(url)
    response = urlopen(url).read()

    currentHash = hashlib.sha224(response).hexdigest()

    lastHash = read_last_hash(fileName)

    if currentHash == lastHash:
        logging.info('No updates to BlueStarMasterList')
        counter = 0

    else:
        logging.info('BlueStarMasterList has been updated')
        save_hash(fileName, currentHash)
        counter = 1

    return counter

def main(url: str, fileName: str):
    updateStatus = webChecker(url, fileName)
    if updateStatus == 0:
        pass
    else:
        subject = 'BlueStarMasterList Update'
        body = """ 
        BlueStarMasterList has been updated 
        """
        sendMail(subject, body)


if __name__ == '__main__':
    dirPath = os.path.dirname(os.path.realpath(__file__))
    fileName = os.path.join(dirPath, 'websiteHash.txt')

    logDirPath = os.path.dirname(os.path.realpath(__file__))
    logFileName = os.path.join(logDirPath, 'websiteLogs.log')

    url = 'https://www.stsci.edu/~bond/GC_blue_stars_master_list.dat'

    logging.basicConfig(filename=logFileName,
                        encoding='utf-8',
                        format='%(levelname)s (%(asctime)s): %(message)s (Line: %(lineno)d [%(filename)s])',
                        datefmt='%d/%m/%Y %I:%M:%S %p',
                        level=logging.INFO)
    main(url, fileName)
