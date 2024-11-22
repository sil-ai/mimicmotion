# Pipeline ClearML

# Import necessary libraries
import os
from clearml import Task, Dataset
from clearml.automation import PipelineController
from dotenv import load_dotenv
import argparse

load_dotenv()

gh_user = os.getenv('GITHUB_USERNAME')
gh_token = os.getenv('GITHUB_TOKEN')


pipe = PipelineController(
     name='pipeline mimicmotion',
     project='MimicMotion',
     version='0.0.2',
     packages=".MimicMotion/requirements.txt",
)

pipe.add_step(
    name='Pipeline mimic',
    base_task_project='MimicMotion',
    base_task_name='Inferencev3',
)

pipe.set_default_execution_queue("jobs_urgent")

pipe.start_locally() # Why this function is active?
