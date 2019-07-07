import urllib2
from os.path import basename
from urlparse import urlsplit
from bs4 import BeautifulSoup 
global urlList
urlList = []

def downloadImages(url):
    
    print url
    global urlList
    if url in urlList:
        return
    urlList.append(url)
    try:
        urlContent = urllib2.urlopen(url).read()
    except:
        return

    soup = BeautifulSoup(''.join(urlContent), 'lxml')
    
    imgTags = soup.findAll('img')
    for imgTag in imgTags:
        imgUrl = imgTag['src']
        try:
            imgData = urllib2.urlopen(imgUrl).read()
            fileName = basename(urlsplit(imgUrl)[2])
            output = open(fileName,'wb')
            output.write(imgData)
            output.close()
        except:
            pass

for i in range (1, 80):
    downloadImages('https://www.zappos.com/men-sneakers-athletic-shoes/CK_XARC81wHAAQLiAgMBAhg.zso?p=' + str(i), 1)