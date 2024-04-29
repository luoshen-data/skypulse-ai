# skypulse-ai

# Project Name

Provide a brief description of what the project does and its purpose.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- [curl](https://curl.se/download.html) (for running test cases)

### Installing

A step-by-step series of examples that tell you how to get a development environment running.

#### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/luoshen-data/skypulse-ai
cd skypulse-ai
```

#### Step 2: Set Up the Conda Environment

Create a Conda environment:

```bash
conda create --name your-env-name python=3.11
```

Activate the Conda environment:

```bash
conda activate your-env-name
```

#### Step 3: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Before running the Flask application, make sure your environment is activated and dependencies are installed.

### Start the Flask Server

To start the server, run:

```bash
export FLASK_APP=app.py
flask run
```

This command will start a development server on http://127.0.0.1:5000/. Adjust `your_application.py` to your actual application entry file name.

## Running the Tests

To test the application, use the following `curl` commands:

### Test Case 1: Non-stop Flights Search

```bash
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d "{\"query\":\"search some non-stop flights from San Francisco to New York City for me on 05/06/2024.\"}"
```

### Test Case 2: Detailed Plan Request

```bash
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d "{\"query\":\"create a reasonable travel plan from San Francisco to New York City for me on 05/15. My email is test@gmail.com\"}"
```
```

Make sure you replace `your_application.py` with the actual entry file for your Flask application. This should effectively guide users on how to start and test the server locally.
