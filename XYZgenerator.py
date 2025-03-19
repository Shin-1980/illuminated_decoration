import json
import numpy as np
import os
import math
import cv2
import glob
from PIL import Image, ImageDraw

class XYZgenerator:
    def __int__(self):
        self.pictFilename = ""
        self.jsonFilename = ""
        self.depthFilename = ""
        self.xyzFilename = ""

        self.depthMap = []    
        self.camIntInv = []
        self.distanceThreshould = 0.0
 
        self.maxJpgCol = 0
        self.maxJpgRow = 0    
        self.maxMapCol = 0
        self.maxMapRow = 0
        self.xScaleJpg2Map = 0
        self.yScaleJpg2Map = 0                

    def setConf(self):
        # Set the dimensions of the image 
        self.maxJpgCol = 1920       # Width of the image in pixels
        self.maxJpgRow = 1440       # Height of the image in pixels
        # Set thresould for filtering noise
        self.distanceThreshould = 0.03

    def getAndConfirmFiles(self, pictFilename: str) -> bool:
        self.pictFilename = pictFilename
        self.xyzFilename = self.pictFilename.replace('.jpeg', '.xyz')
        self.jsonFilename = self.xyzFilename.replace('.xyz', '.json') 
        self.depthFilename = self.xyzFilename.replace('.xyz', '_depth.png')
      
        isSatisfied = True        

        if (os.path.isfile(self.jsonFilename)) == False:
            print(f"Error: {self.jsonFilename} not found.")
            isSatisfied = False        

        if (os.path.isfile(self.depthFilename)) == False:
            print(f"Error: {self.depthFilename} not found.")
            isSatisfied = False        
                
        return isSatisfied

    def calcParam(self) -> bool:
        try:
            with open(self.jsonFilename, 'r') as fr:
                di = json.load(fr)

        except FileNotFoundError:
            print(f"Error:JSON file {self.jsonFilename} not found.")
            return False

        self.depthMap = cv2.imread(self.depthFilename, cv2.IMREAD_UNCHANGED)
        self.depthMap = self.depthMap * 0.001   # change unit from [mm] to [m] 

        self.maxMapCol = max([len(v) for v in self.depthMap])
        self.maxMapRow = len(self.depthMap)

        self.xScaleJpg2Map = int(self.maxJpgCol / self.maxMapCol)
        self.yScaleJpg2Map = int(self.maxJpgRow / self.maxMapRow)                        

        self.camIntInv = np.array(di['cameraIntrinsicsInversed'][0])
        # Row and col are opposite in iOS
        self.camIntInv = self.camIntInv.T
        
        self.localToWorld = np.array(di['localToWorld'][0])    
        # Row and col are opposite in iOS
        self.localToWorld = self.localToWorld.T

        return True

    def generateXYZ(self):

        try:
            img_src = Image.open(self.pictFilename)
        except FileNotFoundError:
            print(f"Error: Image file {self.pictFilename} not found.")
            return
        except IOError:
            print(f"Error: Cannot open image file {self.pictFilename}.")
            return        
        
        img_filtered = img_src.copy()
        draw = ImageDraw.Draw(img_filtered)

        try:
            with open(self.xyzFilename, 'w') as fw:
                for i in range(1, self.maxMapCol - 1):
                    for j in range(1, self.maxMapRow - 1):                        
                        arr_worldPoint = self.transformWorldPoint(i,j)

                        if self.isWithInDistanceThreshould(i, j, arr_worldPoint):    
                            self.writeXYZRGBinNoResolution(fw, img_src, i, j, arr_worldPoint)

        except FileNotFoundError:
            print(f"Error: JSON file {self.xyzFilename} not opened.")
        except IOError:
            print(f"Error: Cannot open image file {self.xyzFilename}.")
            return        

    def extractXYZbasedOnCSV(self, fileToRead: str, fileToWrite: str):

        try:
            img_src = Image.open(self.pictFilename)
        except FileNotFoundError:
            print(f"Error: Image file {self.pictFilename} not found.")
            return
        except IOError:
            print(f"Error: Cannot open image file {self.pictFilename}.")
            return        

        try:
            with open(fileToRead, 'r') as fr:
                print(fileToWrite)
                
                try:
                    with open(fileToWrite, 'w') as fw:
                        header = next(fr)

                        for line in fr:
                            s1,s2 = line.split(",") 
                            x = float(s1)
                            y = float(s2)
                            rgb = img_src.getpixel((x, y))     
                            col = x / self.maxJpgCol * self.maxMapCol
                            row = y / self.maxJpgRow * self.maxMapRow                                
                            arr_worldPoint = self.transformWorldPoint(int(col),int(row))
                            string = f"{arr_worldPoint[0][0]},{arr_worldPoint[1][0]},{arr_worldPoint[2][0]},{rgb[0]},{rgb[1]},{rgb[2]}\n"                        
                            fw.write(string)
        
                except IOError:
                    print(f"Error: {fileToWrite} cannot be opend for writing.")

        except FileNotFoundError:
            print(f"Error: {fileToRead} not opened.")

    def transformWorldPoint(self, i: int, j: int):
        gridPosX = self.maxJpgCol / self.maxMapCol * i
        gridPosY = self.maxJpgRow / self.maxMapRow * j
        depth = self.depthMap[j][i]
        arr_worldPoint = self.calcWorldPoint(gridPosX, gridPosY, depth)

        return arr_worldPoint           

    def calcWorldPoint(self, gridPosX: float, gridPosY: float, depth: float) -> np.ndarray:
        arr_auxData3 = np.array([[gridPosX], [gridPosY], [1]])
        arr_localPoint = self.camIntInv @ arr_auxData3 * depth                      
        arr_auxData4 = np.array([[arr_localPoint[0][0]],[arr_localPoint[1][0]],[arr_localPoint[2][0]],[1]])   
        arr_worldPoint = self.localToWorld @ arr_auxData4                       
        arr_worldPoint = arr_worldPoint / arr_worldPoint[3]        
        
        return arr_worldPoint   

    def calcDistance(self, arr_worldPoint0: np.ndarray, arr_worldPoint1: np.ndarray) -> float:
        distance = (arr_worldPoint1[0] - arr_worldPoint0[0]) ** 2
        distance += (arr_worldPoint1[1] - arr_worldPoint0[1]) ** 2 
        distance += (arr_worldPoint1[2] - arr_worldPoint0[2]) ** 2 
                
        if distance > 0.0:
            distance = np.sqrt(distance)
            
        return distance
    
    def isWithInDistanceThreshould(self, i: int, j: int, arr_worldPoint: np.ndarray) -> bool:
        
        arr_worldPointLeft = self.transformWorldPoint(i-1, j)
        distanceLeft = self.calcDistance(arr_worldPoint, arr_worldPointLeft)

        arr_worldPointRight = self.transformWorldPoint(i+1, j)
        distanceRight = self.calcDistance(arr_worldPoint, arr_worldPointRight)

        arr_worldPointUp = self.transformWorldPoint(i, j-1)
        distanceUp = self.calcDistance(arr_worldPoint, arr_worldPointUp)
        
        arr_worldPointDown = self.transformWorldPoint(i, j+1)
        distanceDown = self.calcDistance(arr_worldPoint, arr_worldPointDown)

        if distanceLeft < self.distanceThreshould or distanceRight < self.distanceThreshould or distanceUp < self.distanceThreshould or distanceDown < self.distanceThreshould:
            return True
                    
        return False    

    def writeXYZRGB(self, fw: 'TextIO', arr_worldPoint: np.ndarray, rgb: tuple[int, int, int]) -> None:
        string = f"{arr_worldPoint[0][0]} {arr_worldPoint[1][0]} {arr_worldPoint[2][0]} {rgb[0]} {rgb[1]} {rgb[2]}\n"
        fw.write(string)
    
    def writeXYZRGBinResolution(self, fw: 'TextIO', img_src: Image.Image, i: int, j: int, arr_worldPoint: np.ndarray):    
        gridPosX = self.maxJpgCol / self.maxMapCol * i
        gridPosY = self.maxJpgRow / self.maxMapRow * j
        depth = self.depthMap[j][i]
        
        offsetX = int(self.xScaleJpg2Map * 0.5)
        offsetY = int(self.yScaleJpg2Map * 0.5)
        
        for x in range(int(gridPosX - offsetX - 1), int(gridPosX + offsetX + 1)):
            for y in range(int(gridPosY - offsetY - 1), int(gridPosY + offsetY + 1)):

                rgb = img_src.getpixel((x, y))                
                
                if x < int(gridPosX):
                    if y < int(gridPosY):
                        offsetDepth = self.depthMap[j-1][i-1] - depth
                    else:
                        offsetDepth = self.depthMap[j+1][i-1] - depth
                else:
                    if y < int(gridPosY):
                        offsetDepth = self.depthMap[j-1][i+1] - depth
                    else:
                        offsetDepth = self.depthMap[j+1][i+1] - depth
                                                            
                if abs(offsetDepth) < 0.005:
                    offsetDepth = offsetDepth * 0.5
                else:
                    offsetDepth = 0

                if x >= int(gridPosX - offsetX * 0.5) and x <= int(gridPosX + offsetX * 0.5):
                    if y >= int(gridPosY - offsetY * 0.5) and y <= int(gridPosY + offsetY * 0.5):
                        offsetDepth = 0

                arr_worldPoint = self.calcWorldPoint(x, y, depth - offsetDepth) 
                self.writeXYZRGB(fw, arr_worldPoint, rgb)

    def writeXYZRGBinNoResolution(self, fw: 'TextIO', img_src: Image.Image, i: int, j: int, arr_worldPoint: np.ndarray):        
        gridPosX = self.maxJpgCol / self.maxMapCol * i
        gridPosY = self.maxJpgRow / self.maxMapRow * j
        depth = self.depthMap[j][i]

        rgb = img_src.getpixel((gridPosX, gridPosY))                

        arr_worldPoint = self.calcWorldPoint(gridPosX, gridPosY, depth) 
        self.writeXYZRGB(fw, arr_worldPoint, rgb)




