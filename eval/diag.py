from evaluate import answer_question, client, CORRECTNESS_PROMPT
from load_golden import load_golden

entry = {e["id"]: e for e in load_golden()}["us-disability-01"]

result = answer_question(entry["question"], entity=entry["entity"],
                         strategy="structure", top_k=15)
answer = result["answer"]

print("QUESTION:", entry["question"])
print("GOLDEN  :", entry["golden_answer"])
print("ANSWER  :", answer)
print("SOURCES :", result["sources"])

# ask the judge again, but print its full reply (verdict + reason), not just the score
user = (f"QUESTION:\n{entry['question']}\n\n"
        f"GOLDEN ANSWER:\n{entry['golden_answer']}\n\n"
        f"SYSTEM ANSWER:\n{answer}")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": CORRECTNESS_PROMPT},
              {"role": "user", "content": user}],
    temperature=0, response_format={"type": "json_object"})
print("JUDGE   :", resp.choices[0].message.content)
