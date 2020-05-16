import json
import os, sys
import time
from urllib import request
import urllib




class ThetaS():

    def __init__(self, v=False):
        
        self.v = v 
        
        # check api version
        try:
            api_version = self._get_api_version()
        except:
            print('---------------------------------------')
            print('Please check the connection with THETA.')
            print('---------------------------------------')
            sys.exit()


        # set API v2.1
        if api_version != 2:
            
            # start session
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name":"camera.startSession"}).encode('ascii')
            res = request.urlopen(url, data)
            sessionId = json.loads(res.read().decode('utf-8'))["results"]["sessionId"]

            # set API v2.1
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name": "camera.setOptions", "parameters": {"sessionId": sessionId, "options": {"clientVersion": 2}}}).encode('ascii')
            res = request.urlopen(url, data)
            
            if v:
                print('set API v2.1')
   

   
    def __del__(self):

        pass

    
    def _get_api_version(self):
        url = 'http://192.168.1.1/osc/state'
        data = b''
        res = request.urlopen(url, data, timeout=5)
        ver = json.loads(res.read().decode('utf-8'))['state']['_apiVersion']
        return ver


    def _get_fingerprint(self):
        url = 'http://192.168.1.1/osc/state'
        data = b''
        res = request.urlopen(url, data)
        fingerprint = json.loads(res.read().decode('utf-8'))["fingerprint"]
        return fingerprint


    def _get_latestFileUrl(self):
        url = 'http://192.168.1.1/osc/state'
        data = b''
        res = request.urlopen(url, data)
        uri = json.loads(res.read().decode('utf-8'))["state"]["_latestFileUrl"]
        return uri

    
    def get_modelname(self):
        url = 'http://192.168.1.1/osc/info'
        res = request.urlopen(url)
        model = json.loads(res.read().decode('utf-8'))['model']
        return model
        
    
    def set_stillmode(self):
        # switch to still image capture mode
        url = 'http://192.168.1.1/osc/commands/execute'
        data = json.dumps({"name":"camera.setOptions",
            "parameters": {"options": {"captureMode": "image"}}}
            ).encode('ascii')
        headers = {'Content-Type': 'application/json'}
        req = request.Request(url=url, data=data, headers=headers)
        res = request.urlopen(req)
        return 


    def take(self):
        
        # record fingerprint before taking photo
        prev_url = self._get_latestFileUrl()
        # take
        url = 'http://192.168.1.1/osc/commands/execute'
        data = json.dumps({"name": "camera.takePicture"}).encode('ascii')
        headers = {'Content-Type': 'application/json'}
        req = request.Request(url=url, data=data, headers=headers)
        res = request.urlopen(req)
        if self.v:
            print('Took a photo')
        
        # get photo url
        while True:
            time.sleep(0.2)
            latest_url = self._get_latestFileUrl()
            if latest_url != prev_url:
                if self.v:
                    print('->', latest_url)
                break
        return latest_url

    
        
    def save(self, fileUri, filename=None):
        
        if filename == None:
            filename = os.path.basename(fileUri)

        if self.v:
            print('Saving a photo')

        content = None
        while content is None:
            try:
                '''
                url = 'http://192.168.1.1/osc/commands/execute'
                data = json.dumps({"name":"camera.getImage", "parameters": {"fileUri": fileUri}}).encode('ascii')
                res = request.urlopen(url, data)
                '''
                res = request.urlopen(fileUri)
                content = res.read()
                
                with open(filename, "wb") as file:
                        file.write(content)
                        
            except urllib.error.HTTPError as err:
                if err.code != 400:
                    print("taken photo may not be saved to the theta storage yet.")
                    raise err
                else:
                    time.sleep(0.2)
        if self.v:
            print('Saved a photo')
            print(filename)

    
    def get_imageSize(self):
        url = 'http://192.168.1.1/osc/commands/execute'
        data = json.dumps({"name":"camera.getOptions",
            "parameters": {"optionNames": ["fileFormat"]}}
            ).encode('ascii')
        headers = {'Content-Type': 'application/json'}
        req = request.Request(url=url, data=data, headers=headers)
        res = request.urlopen(req)
        j = json.loads(res.read().decode('utf-8'))
        H = j['results']['options']['fileFormat']['height']
        W = j['results']['options']['fileFormat']['width']
        return H, W
    

    def get_exposureProgram(self):
        url = 'http://192.168.1.1/osc/commands/execute'
        data = json.dumps({"name":"camera.getOptions",
            "parameters": {"optionNames": ["exposureProgram"]}}
            ).encode('ascii')
        headers = {'Content-Type': 'application/json'}
        req = request.Request(url=url, data=data, headers=headers)
        res = request.urlopen(req)
        j = json.loads(res.read().decode('utf-8'))
        return j['results']['options']['exposureProgram']


    def set_exposureProgram(self, value):
        # value:
        # 1: Manural
        # 2: Auto
        # 3: Aperture priority <--- not supported by THETA S, V
        # 4: shutter speed priority
        # 9: ISO priority
        if value in [1, 2, 4, 9]:
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name":"camera.setOptions",
                "parameters": {"options": {"exposureProgram": value}}}
                ).encode('ascii')
            headers = {'Content-Type': 'application/json'}
            req = request.Request(url=url, data=data, headers=headers)
            res = request.urlopen(req)
            return 0
        else:
            print('Not supported value.')
            return 1
    

    def get_isoSupport(self):
        url = 'http://192.168.1.1/osc/commands/execute'
        data = json.dumps({"name":"camera.getOptions",
            "parameters": {"optionNames": ["isoSupport"]}}
            ).encode('ascii')
        headers = {'Content-Type': 'application/json'}
        req = request.Request(url=url, data=data, headers=headers)
        res = request.urlopen(req)
        j = json.loads(res.read().decode('utf-8'))
        return j['results']['options']['isoSupport']
    
    
    def set_iso(self, value):
        iso_support = self.get_isoSupport()
        if value in iso_support:
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name":"camera.setOptions",
                "parameters": {"options": {"iso": value}}}
                ).encode('ascii')
            headers = {'Content-Type': 'application/json'}
            req = request.Request(url=url, data=data, headers=headers)
            res = request.urlopen(req)
            return 0
        else:
            print('Could not set ISO. Support:')
            print(iso_support)
            return 1
    
    
    def get_shutterSpeedSupport(self):
        url = 'http://192.168.1.1/osc/commands/execute'
        data = json.dumps({"name":"camera.getOptions",
            "parameters": {"optionNames": ["shutterSpeedSupport"]}}
            ).encode('ascii')
        headers = {'Content-Type': 'application/json'}
        req = request.Request(url=url, data=data, headers=headers)
        res = request.urlopen(req)
        j = json.loads(res.read().decode('utf-8'))
        return j['results']['options']['shutterSpeedSupport']
    

    def search_shutterSpeed(self, shutterSpeed):
        support = self.get_shutterSpeedSupport()
        diff = []
        for i in range(len(support)):
            diff.append(abs(support[i] - shutterSpeed))
        i = diff.index(min(diff))
        return support[i]


    def set_shutterSpeed(self, value):
        shutterSpeedSupport = self.get_shutterSpeedSupport()
        if value in shutterSpeedSupport:
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name":"camera.setOptions",
                "parameters": {"options": {"shutterSpeed": value}}}
                ).encode('ascii')
            headers = {'Content-Type': 'application/json'}
            req = request.Request(url=url, data=data, headers=headers)
            res = request.urlopen(req)
            return 0
        else:
            print('Could not set shutterSpeed. Support:')
            print(shutterSpeedSupport)
            return 1
    
    
    def set_whiteBalance(self, value):
        whiteBalanceSupport = [
            'auto', 'daylight', 'shade', 'cloudy-daylight', 'incandescent',
            '_warmWhiteFluorescent', '_dayLightFluorescent',
            '_dayWhiteFluorescent', 'fluorescent', '_bulbFluorescent'
            ]
        
        if value in whiteBalanceSupport:
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name":"camera.setOptions",
                "parameters": {"options": {"whiteBalance": value}}}
                ).encode('ascii')
            headers = {'Content-Type': 'application/json'}
            req = request.Request(url=url, data=data, headers=headers)
            res = request.urlopen(req)
            return 0
        else:
            print('Could not set whiteBalance. Support:')
            print(whiteBalanceSupport)
            return 1


    def set_filter(self, value):
        filterSupport = ['off', 'DR Comp', 'Noise Reduction', 'hdr']
        if value in filterSupport:
            url = 'http://192.168.1.1/osc/commands/execute'
            data = json.dumps({"name":"camera.setOptions",
                "parameters": {"options": {"_filter": value}}}
                ).encode('ascii')
            headers = {'Content-Type': 'application/json'}
            req = request.Request(url=url, data=data, headers=headers)
            res = request.urlopen(req)
            return 0
        else:
            print('Could not set set. Support:')
            print(filterSupport)
            return 1

        
    def auto_adjust_exposure(self, filename, iso=100, EV=0):
        # ISO100 priority, daylight
        self.set_exposureProgram(9) # ISO優先
        self.set_filter('off') # 画像加工フィルタoff
        self.set_iso(iso) # ISO
        self.set_whiteBalance('daylight')
        URI = self.take()
        self.save(URI, filename=filename)
        exif = get_exif(filename)
        s = exif['ExposureTime']
        shutter = s[0] / s[1]
        shutter *= 2 ** EV
        
        # 露出を固定
        self.set_exposureProgram(1) # マニュアル露出
        self.set_filter('off') # 画像加工フィルタoff
        self.set_iso(iso) # ISO
        self.set_whiteBalance('daylight')
        shutter_support = self.search_shutterSpeed(shutter)
        self.set_shutterSpeed(shutter_support)
        return iso, shutter_support


    
    

from PIL import Image
from PIL.ExifTags import TAGS

def get_exif(filename):
    im = Image.open(filename, 'r')
    exif = im._getexif()
    exif_table = {}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        exif_table[tag] = value
    return exif_table




if __name__ == '__main__':
    
    
    theta = ThetaS(v=True)
    print(theta.get_modelname())
    theta.set_stillmode()
    uri = theta.take()
    theta.save(uri)

    '''
    uri_list = []

    uri = theta.take()
    uri_list.append(uri)

    uri = theta.take()
    uri_list.append(uri)

    uri = theta.take()
    uri_list.append(uri)
    
    for i, uri in enumerate(uri_list):
        filename = 'Untitled Export/test%d.jpg' % i
        theta.save(uri, filename)
    ''' 
    
