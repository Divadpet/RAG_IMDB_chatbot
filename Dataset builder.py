import os
from datasets import load_dataset, Dataset
import re
import csv

dataset_name = "BrightData/IMDb-Media"
dataset = load_dataset(dataset_name)
sorted_data = dataset.sort('popularity')
short_data = sorted_data["train"].select(range(25000))

data = []

def preprocess_function(examples):
    if not examples['title']:
        return {}

    title = examples['title']
    
    if examples['storyline']:
        instructions_storyline = [f"Provide the storyline of {title}",
                                  f"What is the story of {title}?",
                                  f"What is the plot of {title}?",
                                  f"What is {title} about?",
                                  f"Summarize the plot of {title}"]
        responses_storyline = [f"The storyline of {title} is: " + examples['storyline']]*len(instructions_storyline)

    if examples['top_cast']:
        instructions_top_cast = [f"List the top cast of {title}",f"What actors were in {title}",f"What is the cast of {title}",f"Who acted in {title}"]
        pattern = r'\"actor\":\"(.*?)\".*?\"character\":\"(.*?)\"'
        matches = re.findall(pattern, examples['top_cast'])
        responses_top_cast = [f"The top cast of {title} is: " + ", ".join([f"{actor}, as {character}" for actor, character in matches])]*len(instructions_top_cast)

    if examples['genres']:
        instructions_genres = [f"What are the genres of {title}?",f"What genre is {title}?",f"What is the genre of {title}?"]
        responses_genres = [examples['genres'].strip('[]').replace('"', '')
        ]*len(instructions_genres)

    if examples['imdb_rating']:
        instructions_imdb_rating = [f"What is the IMDb rating of {title}?",f"what rating does {title} have on IMDB?"]
        responses_imdb_rating = [f"{title} has IMDB rating of " + str(examples['imdb_rating'])]*len(instructions_imdb_rating)
    
    if examples['details_release_date']:
        instructions_release_date = [f"What is the release date of {title}?",f"When was {title} released?"]
        responses_release_date = [f"{title} release date is: " + examples['details_release_date']]*len(instructions_release_date)
    
    if examples['details_countries_of_origin']:
        instructions_countries_of_origin = [f"What are the countries of origin for {title}?",f"What country is {title} from?",f"From what country is {title}?",f"Where did {title} originate?"]
        responses_countries_of_origin = [f"{title} is from: " + examples['details_countries_of_origin']]*len(instructions_countries_of_origin)
    
    if examples['details_language']:
        instructions_language = [f"What languages are used in {title}?"]
        responses_language = [f"The languages spoken in {title} are: " + examples['details_language']]
    
    if examples['details_filming_locations']:
        instructions_filming_locations = [f"Where was {title} filmed?"]
        responses_filming_locations = [f"{title} Was filmed in: " + examples['details_filming_locations']]
    
    if examples['details_production_companies']:
        instructions_production_companies = [f"Which production companies worked on {title}?"]
        responses_production_companies = [f"{title} was produced by: " + examples['details_production_companies']]
    
    if examples['media_type']:
        instructions_media_type = [f"What is the media type for {title}?"]
        responses_media_type = [f"{title} released as a: " + examples['media_type']]
    
    if examples['imdb_rating_count']:
        instructions_imdb_rating_count = [f"What is the IMDb rating count for {title}?",f"How many ratings does {title} have on IMDB?",f"How many people rated {title}?"]
        responses_imdb_rating_count = [f"{title} has: " + str(examples['imdb_rating_count']) + f" ratings on IMDB."]*len(instructions_imdb_rating_count)
    
    if examples['critics_review_count']:
        instructions_critics_review_count = [f"How many critics reviews does {title} have?"]
        responses_critics_review_count = [f"{title} has: " + str(examples['critics_review_count']) + f"critics' reviews on IMDB."]
    
    if examples['review_count']:
        instructions_review_count = [f"How many reviews does {title} have?"]
        responses_review_count = [f"{title} has: " + str(examples['review_count']) + f"reviews on IMDB."]
    
    if examples['review_rating']:
        instructions_review_rating = [f"What is the review rating for {title}?"]
        responses_review_rating = [f"The review rating of {title} is: " + str(examples['review_rating'])]
    
    if examples['featured_review']:
        instructions_featured_review = [f"Provide the featured review for {title}"]
        examples['featured_review'] = examples['featured_review'].replace("&#39;", "'")
        examples['featured_review'] = examples['featured_review'].replace("\n", " ")
        examples['featured_review'] = examples['featured_review'].replace("&quot;", "'")        
        responses_featured_review = [f"Here's a featured review of {title}: " + examples['featured_review']]
    
    if examples['boxoffice_budget']:
        instructions_boxoffice_budget = [f"What is the box office budget for {title}?",f"How much did {title} cost?",f"How much did it cost to make {title}?"]
        responses_boxoffice_budget = [f"{title} cost: " + examples['boxoffice_budget']]*len(instructions_boxoffice_budget)
    
    if examples['popularity']:
        instructions_popularity = [f"What is the popularity score of {title}?",f"What is the IMDB popularity rating of {title} about?",f"How much is {title} popular on IMDB?"]
        responses_popularity = [f"The IMDB popularity score of {title} is: " + str(examples['popularity'])]*len(instructions_popularity)
    
    if examples['presentation']:
        instructions_presentation = [f"What is {title} about?",f"Provide the story of {title} without spoilers",f"Briefly explain the story of {title}"]
        responses_presentation = [f"{title} is about: " + examples['presentation']]*len(instructions_presentation)
    
    if examples['credit']:
        instructions_credits = [f"Provide credits for {title}",f"Who made {title}?",f"Who directed {title}?",f"Who wrote {title}?"]
        pattern = r'\"name\":\"(.*?)\".*?\"title\":\"(.*?)\"'
        matches = re.findall(pattern, examples['credit'])
        responses_credits = [f"The credits of {title} are: " + ", ".join([f"Name: {name}, Title: {title}" for name, title in matches])]*len(instructions_credits)
    
    combined_instructions_responses = []

    if examples['storyline']:
        combined_instructions_responses.extend(zip(instructions_storyline, responses_storyline))

    if examples['top_cast']:
        combined_instructions_responses.extend(zip(instructions_top_cast, responses_top_cast))

    if examples['genres']:
        combined_instructions_responses.extend(zip(instructions_genres, responses_genres))

    if examples['imdb_rating']:
        combined_instructions_responses.extend(zip(instructions_imdb_rating, responses_imdb_rating))

    if examples['details_release_date']:
        combined_instructions_responses.extend(zip(instructions_release_date, responses_release_date))

    if examples['details_countries_of_origin']:
        combined_instructions_responses.extend(zip(instructions_countries_of_origin, responses_countries_of_origin))

    if examples['details_language']:
        combined_instructions_responses.extend(zip(instructions_language, responses_language))

    if examples['details_filming_locations']:
        combined_instructions_responses.extend(zip(instructions_filming_locations, responses_filming_locations))

    if examples['details_production_companies']:
        combined_instructions_responses.extend(zip(instructions_production_companies, responses_production_companies))

    if examples['media_type']:
        combined_instructions_responses.extend(zip(instructions_media_type, responses_media_type))

    if examples['imdb_rating_count']:
        combined_instructions_responses.extend(zip(instructions_imdb_rating_count, responses_imdb_rating_count))

    if examples['critics_review_count']:
        combined_instructions_responses.extend(zip(instructions_critics_review_count, responses_critics_review_count))

    if examples['review_count']:
        combined_instructions_responses.extend(zip(instructions_review_count, responses_review_count))

    if examples['review_rating']:
        combined_instructions_responses.extend(zip(instructions_review_rating, responses_review_rating))

    if examples['featured_review']:
        combined_instructions_responses.extend(zip(instructions_featured_review, responses_featured_review))

    if examples['boxoffice_budget']:
        combined_instructions_responses.extend(zip(instructions_boxoffice_budget, responses_boxoffice_budget))

    if examples['popularity']:
        combined_instructions_responses.extend(zip(instructions_popularity, responses_popularity))

    if examples['presentation']:
        combined_instructions_responses.extend(zip(instructions_presentation, responses_presentation))

    if examples['credit']:
        combined_instructions_responses.extend(zip(instructions_credits, responses_credits))
    
    formatted_instructions_responses = [
        {"question": instruction, "answer": response}
        for instruction, response in combined_instructions_responses
    ]

    data.extend(formatted_instructions_responses)

short_data.map(preprocess_function, remove_columns=short_data.column_names)
'''
with open("dataset2.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["question", "answer"])
    for entry in data:
        writer.writerow([entry["question"], entry["answer"]])
'''
token = ""
repo_id = "Tekarukite/UHK-IMDB-Dataaset2"

new_data = Dataset.from_list(data)

new_data.push_to_hub(repo_id = repo_id,
                  token=token,
                  commit_message="reformated BrightData/IMDb-Media dataset",
                  private=True)

