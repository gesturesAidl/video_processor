# video_processor

## Installation
##### Before Installation

Before starting, make sure you have Python and that it's available from your command line.
You can check this by simply running:

```bash
#!/bin/bash
$ python --version
```

You should get some output like ```3.7.6.```

##### Installation
If you do not have Python, please install the latest 3.x version from python.org or refer to th>


### Install Miniconda

To test your installation, in your terminal window or Anaconda Prompt, run the command conda li>
Open terminal, type "bash" and press ENTER
Then run conda list

### Create Miniconda environment

Execute:

        conda env create -f environment.yml

This will generate the videoprocessor environment with all the required tools installed.
Once created activate the environment by typing:

        conda activate videoprocessor

### Update from environment.yml

    conda env update --file environment.yml
    
### Create .env file
Create a folder with name 'env' inside the video_processor root directory folder and then, create a .env file inside it.
Copy the following code to your .env file and set the fields with your rabbitmq broker connection parameters: 

    RABBIT_USER="..."
    RABBIT_PW="..."
    RABBIT_HOST="..."
    RABBIT_PORT="..."
    VIDEOS_OUT_PATH="..."
    MODEL_DIR="..."
    

### Set the working directory of your app
In your app/app.py file, replace the {abs_path_to_your_project} variable in the 'workdir' var assignment to your absolute path to the project. 

### RUN project
Place yourself in the repository root dir and type: 

    python3 app/app.py
