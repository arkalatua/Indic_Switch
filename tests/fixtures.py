"""Tiny hand-written Hinglish examples for unit tests ONLY (not training data)."""

SENTENCES = [
    ("main kal office nahi ja raha hoon", "HI EN EN HI HI HI HI".split() if False else
     ["HI", "HI", "EN", "HI", "HI", "HI", "HI"]),
    ("yaar this movie was bahut accha", ["HI", "EN", "EN", "EN", "HI", "HI"]),
    ("meeting kab hai ?", ["EN", "HI", "HI", "OTH"]),
    ("please mujhe call karo", ["EN", "HI", "EN", "HI"]),
    ("kya tum free ho tonight", ["HI", "HI", "EN", "HI", "EN"]),
    ("the project deadline is kal", ["EN", "EN", "EN", "EN", "HI"]),
    ("mera laptop kaam nahi kar raha", ["HI", "EN", "HI", "HI", "HI", "HI"]),
    ("i will send the report abhi", ["EN", "EN", "EN", "EN", "EN", "HI"]),
    ("haan theek hai , see you", ["HI", "HI", "HI", "OTH", "EN", "EN"]),
    ("tumhara phone number kya hai", ["HI", "EN", "EN", "HI", "HI"]),
    ("we are going to market kyunki ghar mein kuch nahi hai", ["EN", "EN", "EN", "EN", "EN", "HI", "HI", "HI", "HI", "HI", "HI"]),
    ("actually mujhe nahi pata", ["EN", "HI", "HI", "HI"]),
    ("this is very important , samjhe", ["EN", "EN", "EN", "EN", "OTH", "HI"]),
    ("aaj weather bahut accha hai", ["HI", "EN", "HI", "HI", "HI"]),
    ("can you please help karo", ["EN", "EN", "EN", "EN", "HI"]),
    ("woh bahut busy hai today", ["HI", "HI", "EN", "HI", "EN"]),
]


def rows():
    return [{"tokens": s.split(), "tags": t} for s, t in SENTENCES]
