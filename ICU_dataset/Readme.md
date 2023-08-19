# Emory ICU Dataset


The Emory ICU dataset is provided as a set of parquet files mounted onto the JupyterHub instances you are using.

**Quickstart**: If you have a notebook, look in the `/icu/emory` folder, which has the Emory dataset. The `/icu/mimic` folder has the MIMIC-IV Clinical Database v2.2.


## Guide to work on the Emory ICU Dataset

We have the Emory ICU dataset for. To access the files in the dataset please follow the instructions. 

### Log in to the platform

You have been given your own AWS credentials. Please go to the JupyterHub link provided for your team and input your username and password. The first-time login will require you to provide the old password (which you were given) and change the password with a new one. Don't forget your new password!

### Access the .parquet files

The .parquet files are located in your JupyterHub in the `/icu/` root folder. Once you log in to the JupyterHub, you should open a terminal. Write this commands. 

	cd /icu/

In this directory, there are parquet files and the database file. 

Write- 

	ls 

And you will see all the datatables in .parquet file format. 

### Reading the .parquet files

We have provided an extensive tutorial on using the parquet files in Python. You need to download it from the GitHub! Go back to your home folder:

	cd ~
	wget https://raw.githubusercontent.com/Emory-HITI/datathon/main/ICU_dataset/Tutorial_ICU_Dataset.ipynb

 Open the Tutorial_ICU_Dataset.ipynb notebook and start coding!
