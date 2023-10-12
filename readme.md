### How to run the app in your local server in a browser window:

In the terminal, run the following command to globally install pipenv (for virtual environment): 
```pip install pipenv```

Next, clone this repository and use the terminal to navigate to the folder holding this project. Once there, paste the following in the terminal to install Flask: 
```pipenv install flask```

To activate the virtual environment, paste the following into the terminal: 
```pipenv shell```

To deactivate the virtual environment at some point, type ```exit``` into the terminal.

Run the following commands in the terminal to download packages:
    ```
    pip install pandas
    pip install -U scikit-learn
    python -m pip install -U matplotlib
    pip install numpy
    ```

Paste the following in the terminal to run the app: ```python server.py```
  - This command should give a link to where it is running. Open this link in your browser.
  - To stop this from running, click in the terminal and press CTRL + C
  - It can be run again by typing ```python server.py```, or when done with viewing the project, deactivate the virtual environment by typing ```exit``` in the terminal.
