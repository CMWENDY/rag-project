import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker.
You are given a CLAIM and a SOURCE.
Decide whether the SOURCE directly supports the CLAIM.

Answer "true" only if the source actually contains information that backs up the
claim. If the source is about something else, or does not contain the
information, answer "false".

Respond ONLY as JSON in this exact shape:
{"supported": true or false, "reason": "<one short sentence>"}
"""

def extract_claims(answer):
    tokens = re.split(r'(\[\d+\])', answer)   # keep the [n] tokens in the result
    claims = []
    current_text = ""

    for tok in tokens:
        m = re.fullmatch(r'\[(\d+)\]', tok)
        if m:                                  # this token IS a citation
            claims.append((current_text.strip(), int(m.group(1))))
        elif tok.strip():                      # this token is real text
            current_text = tok                 # start a fresh claim's text
    return claims

def verify_claim(claim, source_text):
    user_prompt = f"CLAIM: {claim}\n\nSOURCE:\n{source_text}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def verify_citations(answer, chunks):
    claims = extract_claims(answer)
    results = []

    for claim_text, n in claims:
        if n < 1 or n > len(chunks):
            results.append({
                "claim": claim_text,
                "citation": n,
                "supported": False,
                "reason": "citation number does not exist",
            })
            continue

        source_chunk = chunks[n - 1]
        verdict = verify_claim(claim_text, source_chunk["text"])

        results.append({
            "claim": claim_text,
            "citation": n,
            "source": source_chunk["source"],
            "supported": verdict["supported"],
            "reason": verdict["reason"],
        })

    return results

if __name__ == "__main__":
    from generator import generate_answer

    query = "what medical and dental benefits does GitLab offer US employees"
    answer, chunks = generate_answer(query)

    print("ANSWER:\n", answer, "\n")

    results = verify_citations(answer, chunks)

    supported = [r for r in results if r["supported"]]
    flagged = [r for r in results if not r["supported"]]

    print(f"Checked {len(results)} citations: "
          f"{len(supported)} supported, {len(flagged)} flagged.\n")

    if flagged:
        print("FLAGGED (unsupported) citations:")
        for r in flagged:
            print(f"  [{r['citation']}] {r['reason']}")
            print(f"      claim: {r['claim']}")
