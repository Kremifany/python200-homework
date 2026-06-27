print("---Task 1: Setup and System Prompt---")
# Load your API key and initialize the client.
# Then define a get_completion() helper function (as seen in the prompt engineering lesson) 
# that takes a messages list and returns the model's text response:
import json
import textwrap
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """
You are "CareerCompass" an experienced job application coach.

WHO YOU HELP:
You assist job seekers who are preparing application materials such as resumes,
cover letters, LinkedIn summaries, and answers to application/interview questions.
Your user may be at any career stage, from entry-level to senior, and may be
changing fields or re-entering the workforce.

WHAT YOU DO:
- Help draft, rewrite, structure, and tighten job application materials.
- Tailor tone and content to a specific role or job description when one is provided.
- Explain WHY you suggest a change so the user learns and can adapt it themselves.

BEHAVIORAL CONSTRAINTS (always follow):
1. Stay strictly focused on job application materials. If the user asks for help with
   something outside this scope (e.g., coding, legal, financial, or personal advice),
   politely decline and steer them back to their application materials.
2. End every full response by reminding the user to carefully REVIEW AND EDIT
   your output, and verify all facts, before submitting it anywhere.
3. Acknowledge that you do NOT know the user's specific company, regional, or industry
   norms. Tell them to apply their own judgment and confirm conventions for their field.
4. Never invent experience, credentials, employers, or accomplishments the user has not
   provided. Ask for missing details instead of fabricating them.

TONE:
Be encouraging, concrete, and concise. Prefer specific, actionable suggestions over
generic advice.
"""

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': "Introduce yourself and tell me how you can help with my job search."},
]
response = get_completion(messages, temperature=1)
print(response)

# Deliberate choice: I assigned the model name role - "CareerCompass" and an
# explicit scope "ONLY job application materials" and refusal rule for
# off-topic requests. Naming the assistant that way scopes its behavior and makes it 
# predictable across the whole project and prevents it from drifting into unrelated advice
# where it could give the user bad guidance.

print("\n---Task 2: Bullet Point Rewriter---\n")
# Write a standalone rewrite_bullets() function that takes a list of resume bullet points and returns improved versions. 
# This function will later be called from inside the chatbot loop.
# Your function should:

# Use delimiters to clearly separate the user's bullet points from your instructions
# Ask for the output as a JSON list where each item has "original" and "improved" keys
# Parse the JSON response and print both versions of each bullet side by side

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)
    print(f"bullet text: {bullet_text}\n")
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with a valid JSON list (a JSON array), no other text,
    and do not wrap the JSON in code fences.
    Each item in the list should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    response = get_completion(messages)

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        print(f"The response is not valid JSON.{response}")
        return []

    # If the model wrapped the list inside a dict (e.g. {"bullets": [...]}),
    # pull out the first value that is actually a list.
    if isinstance(result, dict):
        result = next((v for v in result.values() if isinstance(v, list)), [])

    for item in result:
        print(f"Original: {item['original']}")
        print(f"Improved: {item['improved']}")
        print("-" * 40)

    return result


# Test it with these starter bullets:
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
rewrite_bullets(bullets)

# Are both the original and improved versions printing clearly for each bullet? A: YES
# Do the improvements feel meaningfully better, or are they just rearranged words? A: The improvments are much better
# If the output is weak, try making your prompt more specific about what "strong" looks like. A: The output is great
# 
# OUTPUT:
# Original: Helped customers with their problems
# Improved: Resolved customer inquiries and issues, enhancing satisfaction and loyalty through effective problem-solving.
# ----------------------------------------
# Original: Made reports for the management team
# Improved: Compiled and presented comprehensive reports to the management team, driving informed decision-making and strategicplanning.
# ----------------------------------------
# Original: Worked with a team to finish the project on time
# Improved: Collaborated with a cross-functional team to successfully complete the project two weeks ahead of schedule, ensuring high-quality delivery.
# ----------------------------------------

print("---Task 3: Cover Letter Generator---")
# Write a generate_cover_letter() function that takes a job title and a brief description of the user's background, 
# and returns a cover letter opening paragraph.
# Use few-shot prompting: include at least two examples of strong cover letter openings in your prompt before
# asking for the new one. Your examples should demonstrate 
# the tone and style you want — confident, specific, and not generic.

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences and confident in tone.

    RULES (follow strictly):
    - Every sentence must contain a concrete detail (a real skill, tool, number, or
      experience from the Background). If a sentence would still make sense in someone
      else's cover letter for a different job, rewrite it.
    - NEVER use these clichéd words or phrases (or close variants):
      "innovative solutions", "drive impactful results", "impactful", "results-driven",
      "passionate", "dynamic", "synergy", "leverage", "think outside the box",
      "team player", "go-getter", "hit the ground running", "value add",
      "make a difference", "take it to the next level", "cutting-edge", "fast-paced".
    - Do NOT end with a vague value statement (e.g. "I can contribute to solutions
      that drive results"). Instead, end with ONE specific reason this person fits
      THIS role at [Company], grounded in their actual background.
    - Show, don't tell: instead of saying you are skilled, name the thing you built
      or did that proves it.

    Here are two examples of the style and tone you should match:


    Example 1:
    Role: UX Designer at an education technology company
    Background: Eight years teaching high school English, completed a UX design certificate last year.
    Opening: For eight years I stood in front of thirty teenagers and learned, often the hard way,
    that a lesson only works if it meets people where they actually are. Redesigning my courses
    around how students really learn was my first taste of user-centered design, and a UX
    certificate gave me the vocabulary and tools to do it deliberately. I'm drawn to [Company]
    because building learning products means sweating exactly the details I obsessed over in the
    classroom — clarity, accessibility, and respect for the person on the other side of the screen.

    Example 2:
    Role: Project Manager at a renewable energy firm
    Background: Twelve years as a military logistics officer, recently earned a PMP certification.
    Opening: Coordinating fuel, parts, and people across three time zones with lives on the line
    taught me that a plan is only as good as its weakest handoff. Over twelve years as a logistics
    officer I learned to keep complex operations moving when conditions changed by the hour, and
    earning my PMP gave structure to instincts I'd already been running on. I want to bring that
    discipline to [Company] because scaling clean energy is fundamentally a logistics problem, and
    it's one I've spent my career learning to solve.

    Here is a WEAK opening and a fixed version, so you can see what to avoid:
    Weak (do NOT write like this): "I'm a passionate, results-driven developer eager to
    leverage my skills to contribute to innovative solutions that drive impactful results
    at your company."
    Why it's weak: no concrete detail, full of clichés, could belong to anyone.
    Fixed: "Two years ago I was writing test cases in QA; today I build data pipelines in
    Prefect and Pandas that turn messy CSVs into reports my team actually trusts. I'm
    applying to [Company] because that move from checking other people's code to shipping
    my own is exactly the kind of growth your engineering team is built around."

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    
    response = get_completion(messages)

    

    return response

# Test it with:
job_title = "Junior Software Engineer"
background = "Bachelour degree in Comp Sci, work in QA years ago, working as an intern with react, mongo, fron and backend; recently completed courses Node, React, Pyhton data engeneering\
a Python course and built data pipelines using Prefect and Pandas."

print(f"\n COVER LETTER: \n {generate_cover_letter(job_title, background)}")


# OUTPUT:


#  COVER LETTER:
#  With a Bachelor’s degree in Computer Science and hands-on experience in quality assurance,
#  I’ve honed my problem-solving skills by identifying and resolving bugs that impacted user experience.
#  In my recent internship, I developed user interfaces using React and built data pipelines with Prefect
#  and Pandas, successfully transforming raw data into actionable insights. My completion of courses in
#  Node and Python data engineering has further solidified my technical foundation, allowing me to navigate
#  both front-end and back-end challenges. I am excited about the Junior Software Engineer role at [Company] 
# because I see a perfect alignment between my technical skills 
#  and your commitment to delivering robust software solutions that truly enhance user interactions.

# I chose the teacher-to-UX and military-to-PM examples because they're both career changers,
# like the QA-to-engineer case I'm testing. I wanted the model to see how you take experience from
# one field and connect it to a totally different job, not just copy wording from one industry.
# Putting them in different fields helps with that — the model picks up the pattern (real detail
# from your past, then your new credential, then why this company) instead of repeating "UX words"
# or "PM words" blindly. I also added the weak vs fixed example because telling the model "don't
# be generic" isn't enough; showing a bad opening next to a good one makes the difference clearer.
# Few-shot helps control tone and style because the model copies how the examples sound — confident
# and specific, not like a template cover letter. It also pushes specificity since every example
# sentence has something concrete (years teaching, three time zones, PMP) that couldn't belong
# to someone else.

# A: the output not feels like tailored to specific person just to spesific job title and background as we wanted 
# A: It avoided to use other then prvided credentials
# A: Yes output adapts if the input is changed

print("---Task 4: Moderation Check---")
# Before sending any user input to the model in your chatbot loop, run it through OpenAI's
# moderation endpoint first.

# Write an is_safe(text) function that:
# Calls client.moderations.create() with model="omni-moderation-latest"
# Returns True if the input is not flagged, False if it is
# Prints a short, respectful message if the input is flagged, asking the user to rephrase
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("I'm sorry, but I can't help with that. Please rephrase your request "
              "so we can keep things focused on your job application materials.")
        return False
    return True

# Test your function with at least two inputs — one that should pass and one that should be flagged
#  — and print the result of each test. You want to confirm this is working correctly before wiring
# it into the loop.
print("\nIs it an safe input? Can you help me improve my resume for a marketing role. ->  ")
print(is_safe("Can you help me improve my resume for a marketing role?"))
print("\nIs it an safe input? I am going to hurt the people at that company. ->  " )
print(is_safe("I am going to hurt the people at that company."))

# Before you move on — check:
# Does your flagged test case actually get caught? If not, try a more explicit phrase.
# Does your safe test case pass without triggering any warning?
# What happens if you test a borderline phrase? Look at result.results[0].categories to see which category was triggered.
# Borderline frase like" I dont like people in that company" not flagged

print("---Task 5: The Chatbot Loop---")
# Now assemble everything into a working chatbot.
# Use the starter code below as your structure — your job is to fill in the marked sections.

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        
        
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewrite_bullets(raw_bullets)

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            letter = generate_cover_letter(job_title, background)
            print("\nJob Application Helper:\n")
            print(textwrap.fill(letter, width=80))

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            print(len(messages))
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()



print("---Task 6: Ethics Reflection---")
# Q:Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice?
# Could it favor certain communication styles, industries, or cultural backgrounds?
# A: Because bot was trained on certain more "west" kind of data it can be inclined towards english speaking proffessionals
#  and take a not native speakers as a weak proffessionals.
# Also the culture differences for examples coollective company culture and individualistic culture. 
# so bot can rewrite the resume in different style then the author
# because of the traing data more from IT industry it can incline more to that kind of jobs and not hospitality or nursing let say
# Q: What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
# A: It could hallucinate some information that should be filtered from the response like fabricated facts like procentages that I had in my bullets
# Or the answer could be too generic and might need to be insapected and tighten by human

