# PA2
#### Kornkamol Anasart 20211182

✨This code was written in python ✨
1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
    ```
    git clone https://github.com/bllaka/PA2
    cd PA2
    ```
2.  Create (and activate) a new environment, named `pa2` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n pa2 python=3.6
	source activate pa2
	```
	- __Windows__: 
	```
	conda create --name pa2 python=3.6
	activate pa2
	```
3. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
    ```
    pip install -r requirements.txt
    ```
4. That's it

This implement seperate into 2 parts:
- First part is finding best fundamental matrix with RANSAC. (fundamental_m.py)
- Second part is Triangulation 2D point coordinates to 3D point coordinate. (triangulation.py)
- We already find our best fundamental matrix which was saved in .../saved_data/ in numpy array including:
    - F.nyp = our best fundamental matrix (241 inlier points from 296 possible matched points)
    - inlierp_a.nyp and inlierp_b.nyp = inlier points from RANSAC in both image A and image B (image A is dataset/twoview/sfm01.jpg image B is dataset/twoview/sfm02.jpg
- dataset folder is include all given images
- PLY is in the result folder
