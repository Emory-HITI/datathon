**Emory ICU Dataset** 

****Guide to work on the Emory ICU Dataset****

 

Hello Teams. Welcome to your dataset that you will be using in this datathon to work on your project. 

We have the Emory ICU dataset for. To access the files in the dataset please follow the instructions. 

 

Are you ready to start your journey on Amazon Web Services? 

You have been given your own AWS credentials and Access Keys. Please Go to the link of the AWS console and provide your username and password. The first-time login will require you to provide the old password (which you were given) and change the password with a new one. Once you are done with that, search for S3 and go to S3 console of AWS. There in the S3 console, you will be able to see the buckets of different dataset. You need to find your dataset’s bucket which is “icu-datathon”. The database and required files are found in this bucket.  

 

To get start with the data in your own notebook, follow the steps below. 

 

***Step 1:*** 

Configure your AWS with JupyterHub: 

Now it’s time to configure your JupyterHub with the AWS console where the files are stored. 

Go to your JupyterHub and start a terminal from there.  

In your terminal go to the directory where you want to work. 

Write down the following codes:  

	aws configure 

It will ask for your Access Key ID and Secret Access Key. Give those from the access file that you got previously. 

Then it will ask for the Default region name which you should give as- “us-east-1" and Default output format as- “json”. 

***Step 2:***

Load your data tables or the database in your JupyterHub: 

The .parquet files are located in your JupyterHub. Once you log in to the JupyterHub, you should open a termina. Write this commands. 

	cd .. 

	cd .. 

Now you are in the root directory. Write- 

	cd icu 

In this directory, there are parquet files and the database file. 

Write- 

	ls 

And you will see all the datatables in .parquet file format. 

***Step 3:***

Reading the .parquet files and getting your datatables: 

From your home directory, you can open a Python notebook and start loading neccesary libraries. You can click “Python 3” under the notebook section and start a new notebook. 

Using pandas, you can read the parquet files. For example use the following code to get admissions table. 

	import pandas as pd 
 	admissions=pd.read_parquet('../../icu/emory/admissions.parquet', engine='pyarrow') 

Follow the provided Python Notebook in this directory for more details. 
