import json
import os


def format_results_as_table(results):
    """
    Converts JSON results to a more readable table format.
    Handles upper/lower case and formatting for better display.
    """
    table_data = []
    for row in results:
        new_row = {}
        for key, value in row.items():
            formatted_key = key.title()
            if isinstance(value, dict):
                for k, v in value.items():
                    formatted_sub_key = f"{formatted_key} - {k.title()}"
                    new_row[formatted_sub_key] = v
            else:
                new_row[formatted_key] = value
        table_data.append(new_row)
    return table_data


def generate_sample_questions():
    sample_q_url = os.environ.get("SAMPLE_QUESTIONS")
    with open(sample_q_url, "r") as f:
        sample_questions = json.load(f)
    return sample_questions
