import glob
import copy
import argparse
import os
import open3d as o3d
import numpy as np
import cv2
import colorsys
import json
from XYZgenerator import XYZgenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument(
        "--input", default="input",
        help="image path or directory path which contains images to infer",
    )
    # fmt: on
    args = parser.parse_args()

    return args

class ARFrame2XYZ:                    
    def __init__(self):
        self.pathToFolder = ""
        self.nameOfXYZfile = ""
        self.nameOfFilteredXYZfile = ""

    def setFilename(self, pathToFolder: str):
        self.pathToFolder = pathToFolder
        self.nameOfXYZfile = self.pathToFolder + "/xyz/all.xyz"   
        self.nameOfFilteredXYZfile = self.pathToFolder + "/xyz/filtered.xyz"
        self.pathToNewDirectory = self.pathToFolder + "/xyz"

    def generateXYZ(self):

        filenamesList = glob.glob(self.pathToFolder + '/*.jpeg')

        allNumOfFiles = len(filenamesList)
        curNumofFile = 1

        print("Reading...")

        for pictFilename in filenamesList:
            xyzGen = XYZgenerator()
            xyzGen.setConf()

            if not xyzGen.getAndConfirmFiles(pictFilename) or not xyzGen.calcParam():
                continue

            print(f"{curNumofFile} / {allNumOfFiles} files")
            curNumofFile += 1
            xyzGen.generateXYZ()

        self.mergeXYZ()
    
    def filterNoise(self, brightness_threshold = 0.35):

        try:
            data = np.loadtxt(self.nameOfXYZfile, delimiter=' ')  # Load file
        except FileNotFoundError:
            print(f"Error: File '{self.nameOfXYZfile}' not found.")
            return

        points = data[:, :3]  # XYZ columns
        colors = data[:, 3:6] / 255.0  # Normalize RGB to [0,1] 

        # Convert RGB to HSV and filter by brightness (V)
        hsv_values = np.array([colorsys.rgb_to_hsv(*rgb) for rgb in colors])
        brightness = hsv_values[:, 2]  # Extract V channel

        # Filter points with high brightness
        bright_indices = brightness > brightness_threshold
        bright_points = points[bright_indices]
        bright_colors = colors[bright_indices]

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bright_points)
        pcd.colors = o3d.utility.Vector3dVector(bright_colors)

        # Display the original colored point cloud
        o3d.visualization.draw_geometries([pcd])

        # Downsample with voxel grid
        voxel_size = 0.001  # Adjust voxel size as needed
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Convert Open3D point cloud back to numpy array
        downsampled_points = np.asarray(downsampled_pcd.points)
        downsampled_colors = np.asarray(downsampled_pcd.colors) * 255  # Convert back to 0-255 range

        # Merge XYZ and RGB data for saving
        xyzrgb_data = np.hstack((downsampled_points, downsampled_colors))

        # Save as a new XYZRGB file (space-separated)
        np.savetxt(self.nameOfFilteredXYZfile, xyzrgb_data, fmt="%.6f %.6f %.6f %d %d %d")

    def mergeXYZ(self):

        filenamesList = glob.glob(self.pathToFolder + '/*.xyz')

        os.makedirs(self.pathToNewDirectory, exist_ok=True)
        fileToWrite = self.pathToFolder + '/xyz/all.xyz'

        try:
            with open(fileToWrite, 'w') as fw:
                for xyzFilename in filenamesList:
                    try:
                        with open(xyzFilename, 'r') as fr:
                            for line in fr:
                                fw.write(line)                         
                    except FileNotFoundError:
                        print(f"Error: {xyzFilename} does not exist.")

                    os.remove(xyzFilename)

                print(fileToWrite)

        except IOError:
            print(f"Error: Cannot file {fileToWrite}.")
            return        

    def show_PointCloudsAndFrustum(self, near=0.2, far=0.5):

        def create_camera_frustum(jsonFilename, near: float, far: float):

            try:
                with open(jsonFilename, "r") as file:
                    arframe_data = json.load(file)

            except FileNotFoundError:
                print(f"Error: Cannot file {fileToWrite}.")
                return        

            # Extract necessary parameters correctly
            cameraIntrinsicsInversed = np.array(arframe_data["cameraIntrinsicsInversed"][0])
            localToWorld = np.array(arframe_data["localToWorld"][0])

            # Convert JSON lists to NumPy arrays
            K_inv = np.array(cameraIntrinsicsInversed.T)
            localToWorld = np.array(localToWorld.T)

            # Compute Intrinsic Matrix (K)
            K = np.linalg.inv(K_inv)
            fx, fy = K[0, 0], K[1, 1]  # Focal lengths
            cx, cy = K[0, 2], K[1, 2]  # Principal points

            # Compute Image Size
            width, height = int(2 * cx), int(2 * cy)

            # Define frustum points in CAMERA SPACE
            near_points = np.array([
                [(0 - cx) * near / fx, (0 - cy) * near / fy, near],  # Near top-left
                [(width - cx) * near / fx, (0 - cy) * near / fy, near],  # Near top-right
                [(0 - cx) * near / fx, (height - cy) * near / fy, near],  # Near bottom-left
                [(width - cx) * near / fx, (height - cy) * near / fy, near],  # Near bottom-right
            ])

            far_points = np.array([
                [(0 - cx) * far / fx, (0 - cy) * far / fy, far],  # Far top-left
                [(width - cx) * far / fx, (0 - cy) * far / fy, far],  # Far top-right
                [(0 - cx) * far / fx, (height - cy) * far / fy, far],  # Far bottom-left
                [(width - cx) * far / fx, (height - cy) * far / fy, far],  # Far bottom-right
            ])

            # Camera center
            camera_position = np.array([[0, 0, 0]])

            # Combine all frustum points
            points = np.vstack((camera_position, near_points, far_points))

            # Convert to world coordinates
            points = (localToWorld @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]

            # Define frustum edges
            edges = [
                (0, 1), (0, 2), (0, 3), (0, 4),  # Camera center to near plane
                (1, 2), (2, 4), (4, 3), (3, 1),  # Near plane edges
                (5, 6), (6, 8), (8, 7), (7, 5),  # Far plane edges
                (1, 5), (2, 6), (3, 7), (4, 8),  # Near to far edges
            ]

            # Create Open3D LineSet
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(edges),
            )

            return line_set

        visualization_lists = []

        # Read the file and parse XYZRGB values
        data = np.loadtxt(self.nameOfXYZfile, delimiter=' ')  # Assuming space-separated values
        points = data[:, :3]  # XYZ columns
        colors = data[:, 3:6] / 255.0  # Normalize RGB to [0,1] (if stored in 0-255 format)

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        visualization_lists.append(pcd)

        filenamesList = glob.glob(self.pathToFolder + "/*.json")

        for jsonFile in filenamesList:
            frustum = create_camera_frustum(jsonFile, near, far)
            visualization_lists.append(frustum)
                
        # Display both in the same scene
        o3d.visualization.draw_geometries(visualization_lists)

        visualization_lists = []
        visualization_lists.append(pcd)

def main():
    args = parse_args()
    
    if len(args.input) > 0:
        folder_name = str(args.input)
    else:
        print("Error: The path is incorrect.")
        return

    arf2xyz = ARFrame2XYZ()
    arf2xyz.setFilename(folder_name)
    
    arf2xyz.generateXYZ()
    arf2xyz.show_PointCloudsAndFrustum()

    arf2xyz.filterNoise()

    print("done")

if __name__ == "__main__":
    main()
