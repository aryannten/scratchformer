import os
import sys
import csv
import io
import time
import requests
import random

# Force UTF-8 on Windows stdout/stderr to avoid cp1252 crashes on unicode titles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

HEADERS = {"User-Agent": "ScratchformerDatasetBuilder/2.0 (aryannyadav09@gmail.com)"}

def fetch_csv(url):
    """Download a CSV file from a URL and return a list of dictionaries."""
    print(f"Downloading CSV from {url}...")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    content = response.content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)

def fetch_wikipedia_page(title):
    """Fetch plaintext extract of a Wikipedia page using MediaWiki API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            extract = page_data.get("extract", "")
            if extract and len(extract) > 200:
                print(f"  [OK] Fetched '{title}' ({len(extract):,} chars)")
                return f"\n\n--- {title} ---\n\n" + extract
        print(f"  [Empty] No content found for '{title}'")
    except Exception as e:
        print(f"  [Error] Failed to fetch '{title}': {e}")
    return ""

def clean_player_name(given_name, family_name):
    """Clean and combine player names, dropping 'not applicable' or placeholder strings."""
    given = (given_name or "").strip()
    family = (family_name or "").strip()
    
    # Filter out placeholders
    if given.lower() in ["not applicable", "na", "n/a", "unknown", "none"]:
        given = ""
    if family.lower() in ["not applicable", "na", "n/a", "unknown", "none"]:
        family = ""
        
    full = f"{given} {family}".strip()
    return full if full else "A player"

def clean_text(text):
    """Perform text cleaning to make it suitable for character-level GPT."""
    import re
    # Remove URL encoded template noise and template markers
    text = re.sub(r'%[0-9A-Fa-f]{2}', ' ', text)
    text = re.sub(r'\{+[^}]+\}+', ' ', text)
    # Turn === Section Heading === into Section Heading:
    text = re.sub(r'=+\s*([^=]+?)\s*=+', r'\1.', text)
    
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    
    cleaned = []
    for char in text:
        if ord(char) < 128:
            cleaned.append(char)
        else:
            if char in ["é", "è", "ê", "ë"]: cleaned.append("e")
            elif char in ["á", "à", "â", "ã", "ä", "å"]: cleaned.append("a")
            elif char in ["í", "ì", "î", "ï"]: cleaned.append("i")
            elif char in ["ó", "ò", "ô", "õ", "ö", "ø"]: cleaned.append("o")
            elif char in ["ú", "ù", "û", "ü"]: cleaned.append("u")
            elif char in ["ç", "ć", "č"]: cleaned.append("c")
            elif char in ["ñ", "ń"]: cleaned.append("n")
            elif char in ["š", "ś"]: cleaned.append("s")
            elif char in ["ž", "ź", "ż"]: cleaned.append("z")
            elif char in ["ß"]: cleaned.append("ss")
            else: cleaned.append(" ")
    
    text = "".join(cleaned)
    # Normalize paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    normalized_paras = []
    for p in paras:
        lines = " ".join(line.strip() for line in p.splitlines() if line.strip())
        if len(lines) > 40 and not lines.startswith("--- "):
            normalized_paras.append(lines)
    return "\n\n".join(normalized_paras)

def build_match_narratives(matches, goals, stadiums_dict):
    """Group goals by match and build natural, varied match narrative paragraphs."""
    goals_by_match = {}
    for g in goals:
        mid = g.get('match_id')
        if mid not in goals_by_match:
            goals_by_match[mid] = []
        goals_by_match[mid].append(g)
    
    narratives = []
    
    verbs = [
        "scored for", "found the back of the net for", "netted for",
        "struck for", "scored a crucial goal for", "added a goal for"
    ]
    
    for m in matches:
        mid = m.get('match_id')
        tournament = m.get('tournament_name', 'World Cup')
        stage = m.get('stage_name', 'match')
        date = m.get('match_date', '')
        home = m.get('home_team_name', '')
        away = m.get('away_team_name', '')
        stadium = m.get('stadium_name', '')
        city = m.get('city_name', '')
        score = m.get('score', '')
        home_score = int(m.get('home_team_score') or 0)
        away_score = int(m.get('away_team_score') or 0)
        result = m.get('result', '')
        is_shootout = m.get('penalty_shootout') == '1'
        
        openers = [
            f"On {date}, in the {stage} of the {tournament}, {home} faced {away} at {stadium} in {city}.",
            f"During the {tournament}, {home} and {away} met in the {stage} on {date} at {stadium} in {city}.",
            f"The {stage} fixture of the {tournament} saw {home} take on {away} on {date} in {city}."
        ]
        intro = random.choice(openers)
        
        match_goals = goals_by_match.get(mid, [])
        goal_descriptions = []
        for g in match_goals:
            pname = clean_player_name(g.get('given_name'), g.get('family_name'))
            team = g.get('team_name', '')
            minute = g.get('minute_label', '')
            is_pen = g.get('penalty') == '1'
            is_own = g.get('own_goal') == '1'
            verb = random.choice(verbs)
            
            if is_pen:
                desc = f"{pname} converted a penalty kick for {team} in the {minute} minute"
            elif is_own:
                desc = f"{pname} conceded an own goal in the {minute} minute"
            else:
                desc = f"{pname} {verb} {team} in the {minute} minute"
            goal_descriptions.append(desc)
            
        story = intro
        if goal_descriptions:
            if len(goal_descriptions) == 1:
                story += f" In this match, {goal_descriptions[0]}."
            elif len(goal_descriptions) <= 3:
                story += f" During the game, {', while '.join(goal_descriptions)}."
            else:
                first_part = ", ".join(goal_descriptions[:2])
                second_part = ", and ".join(goal_descriptions[2:])
                story += f" Goals followed in sequence: {first_part}, followed by {second_part}."
                
        if is_shootout:
            home_pens = int(m.get('home_team_score_penalties') or 0)
            away_pens = int(m.get('away_team_score_penalties') or 0)
            pen_winner = home if home_pens > away_pens else away
            story += f" After extra time ended in a {score} draw, {pen_winner} won the penalty shootout {m.get('score_penalties')}."
        elif result == "draw":
            story += f" The contest concluded in a {score} draw."
        else:
            winner = home if home_score > away_score else away
            story += f" The match ended with a final score of {score}, securing a victory for {winner}."
            
        narratives.append(story)
        
    return narratives

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, "custom_corpus.txt")
    
    sections = []
    
    print("=" * 60)
    print("SCRATCHFORMER -- FIFA World Cup Dataset Builder 2.0")
    print("=" * 60)
    
    # ── 1. Fetch Comprehensive Wikipedia Articles ─────────────────────
    print("\n1. Fetching Wikipedia Articles (History, Tournaments, Legends)...")
    wiki_articles = [
        # General & History
        "FIFA World Cup",
        "History of the FIFA World Cup",
        "FIFA World Cup records and statistics",
        "National team appearances in the FIFA World Cup",
        "FIFA World Cup awards",
        
        # Tournaments (1930 to 2026)
        "1930 FIFA World Cup", "1934 FIFA World Cup", "1938 FIFA World Cup",
        "1950 FIFA World Cup", "1954 FIFA World Cup", "1958 FIFA World Cup",
        "1962 FIFA World Cup", "1966 FIFA World Cup", "1970 FIFA World Cup",
        "1974 FIFA World Cup", "1978 FIFA World Cup", "1982 FIFA World Cup",
        "1986 FIFA World Cup", "1990 FIFA World Cup", "1994 FIFA World Cup",
        "1998 FIFA World Cup", "2002 FIFA World Cup", "2006 FIFA World Cup",
        "2010 FIFA World Cup", "2014 FIFA World Cup", "2018 FIFA World Cup",
        "2022 FIFA World Cup", "2026 FIFA World Cup",
        
        # Legendary Finals & Iconic Matches
        "1950 FIFA World Cup final tournament",
        "1954 FIFA World Cup final",
        "1958 FIFA World Cup final",
        "1966 FIFA World Cup final",
        "1970 FIFA World Cup final",
        "1974 FIFA World Cup final",
        "Argentina v England (1986 FIFA World Cup)",
        "1986 FIFA World Cup final",
        "1994 FIFA World Cup final",
        "1998 FIFA World Cup final",
        "2002 FIFA World Cup final",
        "2006 FIFA World Cup final",
        "2010 FIFA World Cup final",
        "Brazil v Germany (2014 FIFA World Cup)",
        "2014 FIFA World Cup final",
        "2018 FIFA World Cup final",
        "2022 FIFA World Cup final",
        
        # Legendary Players & Football Icons
        "Pelé",
        "Diego Maradona",
        "Lionel Messi",
        "Cristiano Ronaldo",
        "Zinedine Zidane",
        "Ronaldo (Brazilian footballer)",
        "Johan Cruyff",
        "Franz Beckenbauer",
        "Gerd Müller",
        "Kylian Mbappé",
        "Miroslav Klose",
        "Luka Modrić",
        "Andrés Iniesta",
        "Ronaldinho",
        
        # Awards & Tactics
        "Total Football",
        "Tiki-taka",
    ]
    
    wiki_texts = []
    for title in wiki_articles:
        text = fetch_wikipedia_page(title)
        if text:
            wiki_texts.append(text)
        time.sleep(0.1)  # polite rate limit
        
    print(f"\nFetched {len(wiki_texts)} Wikipedia articles.")
    sections.extend(wiki_texts)
    
    # ── 2. Fetch CSV Historical Data & Build Narratives ───────────────
    print("\n2. Fetching Historical CSV Datasets...")
    try:
        tournaments = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/tournaments.csv")
        matches = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv")
        goals = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/goals.csv")
        awards = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/award_winners.csv")
        stadiums = fetch_csv("https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/stadiums.csv")
        
        # Tournament summaries
        t_paras = []
        for t in tournaments:
            p = f"The {t['tournament_name']} took place in {t['year']}, proudly hosted by {t['host_country']}. " \
                f"A competitive field of {t['count_teams']} national teams contested the championship. " \
                f"After weeks of competition, {t['winner']} triumphed to lift the trophy."
            t_paras.append(p)
        sections.append("\n\n--- Tournament Overviews ---\n\n" + "\n\n".join(t_paras))
        
        # Stadium summaries
        s_paras = []
        for s in stadiums:
            p = f"The iconic venue {s['stadium_name']} in {s['city_name']}, {s['country_name']} " \
                f"boasts a seating capacity of {s['stadium_capacity']} spectators, hosting historic World Cup encounters."
            s_paras.append(p)
        sections.append("\n\n--- Stadium Guides ---\n\n" + "\n\n".join(s_paras))
        
        # Award winners
        a_paras = []
        for a in awards:
            pname = clean_player_name(a.get('given_name'), a.get('family_name'))
            if pname != "A player":
                p = f"At the {a['tournament_name']}, the prestigious {a['award_name']} was bestowed upon {pname} representing {a['team_name']}."
                a_paras.append(p)
        sections.append("\n\n--- World Cup Honors and Awards ---\n\n" + "\n\n".join(a_paras))
        
        # Match narratives with goals integrated
        print("Synthesizing match narratives...")
        stadiums_dict = {s['stadium_id']: s for s in stadiums}
        match_narratives = build_match_narratives(matches, goals, stadiums_dict)
        sections.append("\n\n--- Historic Match Reports ---\n\n" + "\n\n".join(match_narratives))
        print(f"Synthesized {len(match_narratives)} match narrative reports.")
        
    except Exception as e:
        print(f"Error fetching CSVs: {e}")
        
    # ── 3. Clean and Save Full Corpus ─────────────────────────────────
    print("\n3. Cleaning and assembling final corpus...")
    full_text = "\n\n".join(sections)
    cleaned_corpus = clean_text(full_text)
    
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(cleaned_corpus)
        
    file_size_kb = os.path.getsize(dest_path) / 1024
    print(f"\n[OK] SUCCESS! Dataset saved to {dest_path}")
    print(f"  Total size: {file_size_kb:,.1f} KB ({file_size_kb/1024:.2f} MB)")
    print(f"  Total characters: {len(cleaned_corpus):,}")
    print(f"  Total paragraphs: {cleaned_corpus.count(chr(10)+chr(10)):,}")

if __name__ == "__main__":
    main()

