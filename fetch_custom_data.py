import os
import csv
import io
import requests

def fetch_csv(url):
    """Download a CSV file from a URL and return a list of dictionaries."""
    headers = {"User-Agent": "ScratchformerDatasetBuilder/1.0 (aryannyadav09@gmail.com)"}
    print(f"Downloading CSV from {url}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    # Decode content
    content = response.content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)

def fetch_wikipedia_page(title):
    """Fetch plaintext extract of a Wikipedia page using MediaWiki API."""
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "ScratchformerDatasetBuilder/1.0 (aryannyadav09@gmail.com)"}
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title
    }
    print(f"Fetching Wikipedia article: '{title}'...")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        extract = page_data.get("extract", "")
        if extract:
            return f"\n\n--- Wikipedia Article: {title} ---\n\n" + extract
    return ""

def clean_text(text):
    """Perform basic text cleaning to make it suitable for character-level GPT."""
    # Replace curly quotes and apostrophes with standard ones
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-") # En-dash and em-dash to regular hyphen
    # Replace other problematic non-ASCII characters or symbols
    # Keep standard letters, digits, punctuation, and whitespace
    cleaned = []
    for char in text:
        if ord(char) < 128:
            cleaned.append(char)
        else:
            # Fallback to standard chars where appropriate
            if char in ["é", "è", "ê"]: cleaned.append("e")
            elif char in ["á", "à", "â", "ã"]: cleaned.append("a")
            elif char in ["í", "ì", "î"]: cleaned.append("i")
            elif char in ["ó", "ò", "ô", "õ", "ö"]: cleaned.append("o")
            elif char in ["ú", "ù", "û", "ü"]: cleaned.append("u")
            elif char in ["ç"]: cleaned.append("c")
            elif char in ["ñ"]: cleaned.append("n")
            elif char in ["ß"]: cleaned.append("ss")
            else: cleaned.append(" ") # replace other non-ascii with spaces
    
    text = "".join(cleaned)
    # Normalize multiple whitespace/newlines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text

def get_full_name(given_name, family_name):
    """Combine given name and family name cleanly."""
    name = f"{given_name} {family_name}".strip()
    return name

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, "custom_corpus.txt")
    
    sentences = []
    
    # 1. Fetch Tournaments
    try:
        tournaments = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/tournaments.csv")
        print(f"Loaded {len(tournaments)} tournaments.")
        for row in tournaments:
            s = f"The {row['tournament_name']} was held in {row['year']} and hosted by {row['host_country']}. " \
                f"The tournament was won by {row['winner']}. " \
                f"A total of {row['count_teams']} teams participated in this tournament."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching tournaments: {e}")

    # 2. Fetch Stadiums
    try:
        stadiums = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/stadiums.csv")
        print(f"Loaded {len(stadiums)} stadiums.")
        for row in stadiums:
            s = f"The stadium {row['stadium_name']} is located in {row['city_name']}, {row['country_name']} and has a capacity of {row['stadium_capacity']} spectators."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching stadiums: {e}")

    # 3. Fetch Matches
    try:
        matches = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv")
        print(f"Loaded {len(matches)} matches.")
        for row in matches:
            s = f"On {row['match_date']}, in the {row['stage_name']} of the {row['tournament_name']}, {row['home_team_name']} played against {row['away_team_name']} at {row['stadium_name']} in {row['city_name']}. " \
                f"The final score was {row['score']}. The home team scored {row['home_team_score']} and the away team scored {row['away_team_score']}. " \
                f"The official result of the match was a {row['result']}."
            if row.get('penalty_shootout') == '1':
                s += f" The match went to a penalty shootout, which ended {row['score_penalties']} with {row['home_team_score_penalties']} goals to {row['away_team_score_penalties']}."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching matches: {e}")

    # 4. Fetch Goals
    try:
        goals = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/goals.csv")
        print(f"Loaded {len(goals)} goals.")
        for row in goals:
            player_name = get_full_name(row.get('given_name', ''), row.get('family_name', ''))
            s = f"In the {row['tournament_name']} match {row['match_name']} on {row['match_date']}, {player_name} scored a goal for {row['team_name']} in the {row['minute_label']} minute of the match."
            if row.get('penalty') == '1':
                s += " The goal was scored from a penalty kick."
            if row.get('own_goal') == '1':
                s += " This was an own goal."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching goals: {e}")

    # 5. Fetch Award Winners
    try:
        awards = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/award_winners.csv")
        print(f"Loaded {len(awards)} award winners.")
        for row in awards:
            player_name = get_full_name(row.get('given_name', ''), row.get('family_name', ''))
            s = f"At the {row['tournament_name']}, the {row['award_name']} award was won by {player_name} of {row['team_name']}."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching award winners: {e}")

    # 6. Fetch Penalty Kicks
    try:
        penalties = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/penalty_kicks.csv")
        print(f"Loaded {len(penalties)} penalty shootout kicks.")
        for row in penalties:
            player_name = get_full_name(row.get('given_name', ''), row.get('family_name', ''))
            outcome = "converted" if row.get('converted') == '1' else "missed"
            s = f"In the penalty shootout of the {row['tournament_name']} match {row['match_name']}, {player_name} of {row['team_name']} took a penalty kick and {outcome} the shot."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching penalty kicks: {e}")

    # 7. Fetch Referee Appearances
    try:
        referees = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/referee_appearances.csv")
        print(f"Loaded {len(referees)} referee appearances.")
        for row in referees:
            referee_name = get_full_name(row.get('given_name', ''), row.get('family_name', ''))
            s = f"In the {row['tournament_name']} match {row['match_name']}, the match referee was {referee_name} representing {row['country_name']}."
            sentences.append(s)
    except Exception as e:
        print(f"Error fetching referee appearances: {e}")

    # Combine sentences
    sentence_corpus = "\n".join(sentences)
    
    # 8. Fetch Wikipedia pages for 2026 World Cup
    wiki_titles = [
        "2026 FIFA World Cup",
        "2026 FIFA World Cup qualification",
        "2026 FIFA World Cup venues"
    ]
    wiki_text = ""
    for title in wiki_titles:
        try:
            wiki_text += fetch_wikipedia_page(title)
        except Exception as e:
            print(f"Error fetching Wikipedia page '{title}': {e}")
            
    # Combine everything and clean
    full_corpus = sentence_corpus + wiki_text
    print("Cleaning text and writing output...")
    cleaned_corpus = clean_text(full_corpus)
    
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(cleaned_corpus)
        
    print(f"Success! Custom dataset written to {dest_path}")
    print(f"File size: {os.path.getsize(dest_path)/1024:.1f} KB")

if __name__ == "__main__":
    main()
