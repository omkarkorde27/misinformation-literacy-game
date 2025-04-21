import csv
import random
from datetime import datetime, timedelta
import uuid

# Function to generate fake news dataset
def generate_fake_news_dataset(num_entries=500, output_file="fake_news_dataset.csv"):
    # Categories of fake news
    categories = [
        "health", "politics", "disaster", "technology", "science", 
        "entertainment", "business", "education", "environment", 
        "sports", "finance", "security", "space", "celebrity"
    ]
    
    # Types of fake content sources
    source_types = ["fake_tweet", "fake_headline", "fake_article", "fake_social", "fake_blog"]
    
    # Common misinformation narrative patterns
    narrative_templates = [
        "BREAKING: {entity} discovers that {item} {action} {percentage}% of {condition}, {authority} suppressing evidence",
        "{disaster} Destroys {location}! {consequence}. Government covering up actual {metric}.",
        "EXCLUSIVE: {person} announces surprise {action}, cites '{bizarre_reason}' as reason",
        "New study finds {common_item} emit {harmful_thing} that causes {health_issue} in {vulnerable_group}",
        "LEAKED DOCUMENT: {organization} secretly {nefarious_action} through {medium}, {consequence}",
        "{authority} to implement mandatory {controversial_policy} starting {future_date}",
        "Famous {celebrity_type} {criminal_action} for {crime}, faces {severe_consequence}",
        "Scientists confirm {simple_solution} cures {serious_condition} in {high_percentage}% of patients",
        "New law allows {authority} to {punitive_action} for {common_activity}",
        "CONFIRMED: {entity} found to be using {unethical_practice} in {location}",
        "ALERT: {product} found to contain {dangerous_substance} linked to {disease}",
        "INSIDER REPORT: {company} to replace {percentage}% of workforce with AI by {year}",
        "{politician} secretly met with {controversial_figure} to discuss {conspiracy_topic}",
        "Doctors won't tell you: {common_food} can eliminate {health_condition} overnight",
        "{country} developing secret {weapon_type} capable of {exaggerated_capability}",
        "Economic COLLAPSE imminent as {financial_entity} prepares to {financial_action}",
        "{celebrity} reveals shocking truth about {industry}: '{sensational_quote}'",
        "Anonymous whistleblower: {organization} has been {covering_up} for decades",
        "Scientists baffled by {unexplainable_phenomenon} appearing in {location}",
        "URGENT: {everyday_product} recalled after being linked to {severe_consequence}"
    ]
    
    # Components to fill in templates
    components = {
        "entity": ["Scientists", "Researchers", "Expert panel", "Government agency", "Tech giant", 
                  "Medical team", "Secret group", "Whistleblower", "Anonymous source", "Investigative committee"],
        "person": ["Prime Minister", "President", "CEO", "Famous scientist", "Celebrity", 
                  "Sports star", "Religious leader", "Tech billionaire", "Political leader", "Military general"],
        "politician": ["Senior minister", "Opposition leader", "Mayor", "Governor", "Senator", 
                      "MP", "President", "Prime Minister", "Party chief", "Cabinet member"],
        "item": ["coffee", "tea", "vitamin D", "common herb", "household spice", "electronic device", 
                "smartphone app", "wearable device", "specific food", "everyday activity"],
        "common_food": ["turmeric", "apple cider vinegar", "garlic", "honey", "coconut oil", 
                       "lemon water", "cinnamon", "baking soda", "olive oil", "ginger"],
        "action": ["prevents", "accelerates", "triggers", "eliminates", "reverses", 
                  "causes", "disrupts", "transforms", "destroys", "manipulates"],
        "percentage": [95, 99, 87, 92, 78, 83, 91, 97, 85, 93],
        "high_percentage": [91, 94, 97, 99, 92, 95, 89, 98, 93, 96],
        "condition": ["COVID cases", "cancer risk", "aging process", "memory loss", "fertility", 
                     "immune response", "brain function", "chronic pain", "genetic disorders", "viral infections"],
        "authority": ["WHO", "Government", "Health ministry", "United Nations", "International agency", 
                     "Federal authorities", "Regulatory body", "Central committee", "Global consortium", "State officials"],
        "disaster": ["Earthquake", "Tornado", "Flood", "Meteor impact", "Solar flare", 
                    "Volcanic eruption", "Tsunami", "Gas explosion", "Chemical leak", "Nuclear incident"],
        "location": ["major city", "famous landmark", "government building", "international airport", "popular tourist destination", 
                    "major university", "tech headquarters", "financial district", "historic site", "strategic facility"],
        "consequence": ["Thousands trapped inside", "Millions affected", "Economy in ruins", "Infrastructure collapsed", 
                       "Communication networks down", "Emergency services overwhelmed", "Critical shortage of supplies", 
                       "Massive evacuation ordered", "International aid requested", "State of emergency declared"],
        "metric": ["death toll", "damage costs", "radiation levels", "casualty figures", 
                  "economic impact", "affected population", "rescue efforts", "international response", 
                  "environmental damage", "recovery timeline"],
        "bizarre_reason": ["alien contact", "time travel experience", "mind control discovery", "parallel universe evidence", 
                          "spiritual awakening", "supernatural encounter", "secret technology", "conspiracy revelation", 
                          "ancient prophecy", "health conspiracy"],
        "common_item": ["household plants", "smart devices", "popular appliances", "children's toys", "cosmetic products", 
                       "pet food", "bottled water", "cleaning supplies", "packaged foods", "furniture"],
        "harmful_thing": ["radiation", "toxic chemicals", "harmful frequencies", "dangerous compounds", "behavioral manipulation signals", 
                         "mind-altering waves", "carcinogenic particles", "addictive substances", "genetic modifiers", "nanobots"],
        "health_issue": ["memory loss", "hormonal changes", "developmental delays", "organ failure", "behavioral disorders", 
                        "cognitive decline", "immune suppression", "genetic mutations", "chronic illnesses", "psychological dependency"],
        "vulnerable_group": ["children", "elderly", "pregnant women", "teenagers", "immunocompromised people", 
                           "specific ethnic groups", "people with certain blood types", "genetically predisposed individuals", 
                           "urban populations", "rural communities"],
        "organization": ["Major tech company", "Government agency", "Multinational corporation", "Social media platform", 
                        "Pharmaceutical giant", "Banking consortium", "Media conglomerate", "Retail chain", 
                        "Telecommunications provider", "International organization"],
        "nefarious_action": ["recording all conversations", "tracking users", "manipulating emotions", "collecting DNA", 
                            "mining personal data", "conducting experiments", "implanting thoughts", "testing experimental technology", 
                            "altering memories", "influencing decisions"],
        "medium": ["smart speakers", "mobile phones", "vaccination programs", "public water supply", "popular websites", 
                  "TV broadcasts", "food additives", "social media algorithms", "wearable technology", "banking systems"],
        "controversial_policy": ["microchipping", "social credit system", "thought monitoring", "mandatory medication", 
                               "restricted travel", "internet censorship", "compulsory DNA collection", "financial tracking", 
                               "behavioral modification", "digital identity"],
        "future_date": ["May 2025", "late 2025", "early 2026", "July 2025", "next quarter", 
                       "next fiscal year", "coming months", "new year", "next administration", "within weeks"],
        "celebrity_type": ["actor", "musician", "athlete", "influencer", "TV personality", 
                         "politician", "tech CEO", "royal family member", "director", "fashion icon"],
        "criminal_action": ["arrested", "caught", "detained", "investigated", "charged", 
                          "implicated", "exposed", "indicted", "sentenced", "prosecuted"],
        "crime": ["smuggling exotic animals", "tax evasion", "insider trading", "identity theft scheme", 
                 "cryptocurrency fraud", "selling classified information", "environmental violations", 
                 "intellectual property theft", "bribery scandal", "illegal surveillance"],
        "severe_consequence": ["30 years in prison", "billions in fines", "lifetime ban", "international sanctions", 
                             "complete asset seizure", "loss of license", "mandatory public apology", 
                             "community service", "house arrest", "extradition"],
        "simple_solution": ["drinking lemon water with baking soda", "specific vitamin combination", "ancient remedy", 
                          "special exercise routine", "common household item", "specific breathing technique", 
                          "unusual diet", "alternative therapy", "specific sleeping position", "digital detox"],
        "serious_condition": ["diabetes", "cancer", "heart disease", "chronic pain", "autoimmune disorders", 
                           "depression", "anxiety", "Alzheimer's", "arthritis", "obesity"],
        "punitive_action": ["fine citizens", "monitor", "restrict access", "impose curfew", "mandatory registration", 
                          "revoke privileges", "increase taxes", "public shaming", "detention", "community service"],
        "common_activity": ["using more than 50 liters of water per day", "exceeding online time limits", 
                          "posting certain content", "traveling between regions", "certain purchasing patterns", 
                          "growing specific plants", "owning specific items", "using non-approved products", 
                          "exceeding carbon limits", "speaking against policies"],
        "unethical_practice": ["forced labor", "child labor", "illegal surveillance", "human experiments", 
                             "toxic waste dumping", "tax avoidance schemes", "price fixing", "psychological manipulation", 
                             "discriminatory practices", "wildlife trafficking"],
        "product": ["Popular smartphone", "Children's vitamin", "Breakfast cereal", "Bottled water", "Fast food chain", 
                   "Household cleaner", "Baby formula", "Skincare product", "Pain medication", "Pet food"],
        "dangerous_substance": ["carcinogen", "neurotoxin", "banned chemical", "radioactive material", "harmful additive", 
                              "illegal compound", "synthetic hormone", "undisclosed drug", "industrial solvent", "mind-altering substance"],
        "disease": ["rare cancer", "neurological disorder", "reproductive harm", "immune deficiency", "liver damage", 
                   "kidney failure", "birth defects", "blood disorder", "cognitive decline", "genetic mutation"],
        "company": ["Tech giant", "Banking corporation", "Manufacturing conglomerate", "Retail empire", "Healthcare provider", 
                   "Media company", "Fast food chain", "Automobile manufacturer", "Pharmaceutical corporation", "Energy company"],
        "year": [2025, 2026, 2027, 2028, 2029],
        "controversial_figure": ["foreign agent", "conspiracy theorist", "sanctioned individual", "extremist leader", 
                               "disgraced executive", "infamous hacker", "cult leader", "alleged criminal", 
                               "banned journalist", "radical activist"],
        "conspiracy_topic": ["surveillance program", "election interference", "population control", "currency manipulation", 
                           "weather modification", "censorship initiative", "false flag operations", "mind control technology", 
                           "classified weapon system", "secret space program"],
        "financial_entity": ["Central bank", "Stock market", "Major investment firm", "International monetary fund", 
                           "National treasury", "Credit institution", "Global financial system", "Cryptocurrency exchange", 
                           "Pension funds", "Sovereign wealth fund"],
        "financial_action": ["collapse the currency", "freeze all assets", "implement digital currency", "cancel all debts", 
                           "seize private savings", "restrict withdrawals", "manipulate interest rates", "initiate market crash", 
                           "default on obligations", "impose capital controls"],
        "celebrity": ["A-list actor", "Famous musician", "Top athlete", "Royal family member", "Tech billionaire", 
                     "Reality TV star", "Influential politician", "Popular influencer", "Award-winning director", "Fashion icon"],
        "industry": ["entertainment industry", "pharmaceutical business", "government operations", "tech giants", 
                    "banking system", "food industry", "fashion world", "sports business", "media empires", "health sector"],
        "sensational_quote": ["Everything you know is a lie", "They've been controlling us for years", 
                            "The truth will shock the world", "No one is ready for what's coming", 
                            "This corruption goes to the highest levels", "The evidence was destroyed", 
                            "I've seen what they're hiding", "People would revolt if they knew", 
                            "It's all about to collapse", "The cover-up is massive"],
        "organization": ["Government agency", "Major corporation", "Medical establishment", "Military complex", 
                        "Educational system", "Religious institution", "Scientific community", "Banking industry", 
                        "Tech conglomerate", "Media organization"],
        "covering_up": ["covering up critical evidence", "hiding breakthrough cures", "suppressing alternative energy", 
                       "concealing extraterrestrial contact", "manipulating statistical data", "hiding economic collapse", 
                       "withholding crucial health information", "distorting historical facts", 
                       "concealing environmental damage", "hiding surveillance technologies"],
        "unexplainable_phenomenon": ["strange light patterns", "unexplained animal behavior", "mysterious health symptoms", 
                                    "unusual electronic disturbances", "unidentified flying objects", "bizarre weather patterns", 
                                    "unexplained disappearances", "mysterious structures", "anomalous radiation", "time distortions"],
        "everyday_product": ["Popular pain reliever", "Children's toy", "Smartphone model", "Car model", "Baby formula", 
                           "Common appliance", "Widely used app", "Bottled water brand", "Fast food item", "Household cleaner"],
        "country": ["Foreign adversary", "Emerging superpower", "Neighboring nation", "Technological leader", 
                   "Military power", "Regional ally", "Nuclear-capable state", "Economic competitor", 
                   "Hostile regime", "Developing nation"],
        "weapon_type": ["satellite weapon", "climate control device", "mind-influencing technology", 
                       "biological agent", "drone swarm", "artificial intelligence", "sonic weapon", 
                       "electromagnetic pulse device", "nuclear technology", "autonomous system"],
        "exaggerated_capability": ["controlling weather patterns", "disrupting global communications", 
                                  "remote mind control", "causing natural disasters", "manipulating financial markets", 
                                  "disabling military systems", "instant surveillance anywhere", 
                                  "manipulating elections globally", "triggering health crises", "altering genetic code remotely"]
    }
    
    # Explanations of why the content is fake
    explanation_templates = [
        "Misrepresents scientific research; no credible studies support this claim. Uses conspiratorial framing.",
        "No such event occurred on this date. Image circulated was from a different incident.",
        "Completely fabricated story with no basis in reality. No such announcement was made.",
        "No scientific evidence supports this claim. Study cited doesn't exist.",
        "Misrepresents how {technology} works. Document cited is fabricated.",
        "No such policy has been announced or is being planned. Classic conspiracy theory narrative.",
        "No such incident occurred. Image used was from an unrelated event.",
        "Dangerous health misinformation. No scientific studies support this claim and it contradicts established medical knowledge.",
        "No such law has been proposed or passed. Uses {concern} to spread false information.",
        "Accusations not supported by evidence. No investigations have found such practices.",
        "Presents correlation as causation. Cherry-picks data while ignoring contradictory evidence.",
        "Uses emotionally charged language and anonymous sources that cannot be verified.",
        "Misquotes public figure. Original statement was taken out of context.",
        "Classic fearmongering using technical jargon to sound credible while making impossible claims.",
        "Presents common {phenomenon} as unusual or dangerous without scientific basis."
    ]
    
    # Technology and concerns for explanation templates
    explanation_components = {
        "technology": ["smartphone technology", "cloud storage", "encryption", "artificial intelligence", 
                       "machine learning", "voice recognition", "facial recognition", "GPS tracking", 
                       "social media algorithms", "data processing"],
        "concern": ["environmental concerns", "public health fears", "economic anxiety", "national security worries", 
                   "privacy concerns", "technological fear", "political division", "social unrest", 
                   "cultural tensions", "religious sentiments"],
        "phenomenon": ["side effect", "natural process", "statistical variation", "common symptom", 
                      "expected outcome", "normal reaction", "standard procedure", "routine occurrence", 
                      "typical pattern", "regular fluctuation"]
    }
    
    # Generate fact-check domain and path
    fact_check_domains = ["factcheck.example.org", "truthmeter.example.com", "verifynews.example.net", 
                          "debunker.example.org", "realitychecker.example.com"]
    
    # Start and end dates for the publication dates
    start_date = datetime(2025, 3, 15)
    end_date = datetime(2025, 4, 15)
    
    # Generate the fake news entries
    fake_news_entries = []
    
    for _ in range(num_entries):
        # Generate random date within range
        days_range = (end_date - start_date).days
        random_days = random.randint(0, days_range)
        publication_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # Select random category and source type
        category = random.choice(categories)
        source_type = random.choice(source_types)
        
        # Generate content by selecting a template and filling it
        template = random.choice(narrative_templates)
        
        # Replace template placeholders with random selections from components
        for placeholder in components:
            if "{" + placeholder + "}" in template:
                replacement = random.choice(components[placeholder])
                template = template.replace("{" + placeholder + "}", str(replacement))
        
        # Generate explanation
        explanation_template = random.choice(explanation_templates)
        for placeholder in explanation_components:
            if "{" + placeholder + "}" in explanation_template:
                replacement = random.choice(explanation_components[placeholder])
                explanation_template = explanation_template.replace("{" + placeholder + "}", str(replacement))
        
        # Generate fake fact-check URL
        domain = random.choice(fact_check_domains)
        path = "-".join(template.split()[:3]).lower().replace(":", "").replace("!", "").replace(".", "").replace(",", "")
        fact_check_url = f"https://{domain}/{path}"
        
        # Generate unique ID
        unique_id = str(uuid.uuid4())[:8]
        
        # Add entry to list
        fake_news_entries.append({
            "id": unique_id,
            "content": template,
            "source_type": source_type,
            "publication_date": publication_date,
            "category": category,
            "fact_check_url": fact_check_url,
            "explanation": explanation_template
        })
    
    # Write to CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["id", "content", "source_type", "publication_date", "category", "fact_check_url", "explanation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in fake_news_entries:
            writer.writerow(entry)
    
    print(f"Generated {num_entries} fake news entries and saved to {output_file}")
    return output_file

# Generate the dataset
if __name__ == "__main__":
    filename = generate_fake_news_dataset(500, "fake_news_dataset_500.csv")
    print(f"Dataset created successfully: {filename}")