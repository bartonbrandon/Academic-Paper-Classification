import textstat
import os
import pandas as pd


def word_count(test_data):
    return len(test_data.split())


def applied_vs_theoretical(test_data, keywords):
    applied_words = [x for x in list(keywords["Applied"]) if x == x]
    theory_words = [x for x in list(keywords["Theoretical"]) if x == x]
    applied_words_found = 0
    theory_words_found = 0
    for k in applied_words:
        applied_words_found += test_data.count(k)
    for k in theory_words:
        theory_words_found += test_data.count(k)
    if applied_words_found + theory_words_found == 0:
        return [applied_words_found, theory_words_found, 0]
    else:
        return [applied_words_found, theory_words_found, applied_words_found/(applied_words_found+theory_words_found)]


def readFile():
    word = "novel"
    keywords = pd.read_excel("keywords.xlsx")
    df = pd.DataFrame(columns=[
        "File",
        # "Word Count"
        "Flesch Reading Ease",
        "Smog Index",
        "Flesch Kincaid Grade",
        "Coleman Liau Index",
        "Automated Readability Index",
        "Dale Chall Readability Score",
        "Difficult Words",
        "Linear Write Formula",
        "Gunning Fog",
        "Text Standard",
        "Fernandez Huerta",
        "Szigriszt Pazos",
        "Gutierrez Polini",
        "Crawford",
        "Applied words found",
        "Theory words found",
        "Applied score",
        f"Frequency of '{word}'"
    ])
    for infile in os.listdir("papers"):
        with open("papers/" + infile, "r") as file:
            test_data = file.read().replace("\n", "")
            print("counting...")
            count = test_data.count(word)
            wcount = word_count(test_data)
            avt = applied_vs_theoretical(test_data, keywords)
            row = [
                infile,
                textstat.flesch_reading_ease(test_data),
                textstat.smog_index(test_data),
                textstat.flesch_kincaid_grade(test_data),
                textstat.coleman_liau_index(test_data),
                textstat.automated_readability_index(test_data),
                textstat.dale_chall_readability_score(test_data),
                textstat.difficult_words(test_data),
                textstat.linsear_write_formula(test_data),
                textstat.gunning_fog(test_data),
                textstat.text_standard(test_data),
                textstat.fernandez_huerta(test_data),
                textstat.szigriszt_pazos(test_data),
                textstat.gutierrez_polini(test_data),
                textstat.crawford(test_data),
                avt[0],
                avt[1],
                avt[2],
                count
            ]
            df.loc[len(df.index)] = row
        file.close()
    df.to_excel("outputs.xlsx")


def add_fk_scores(df):
    abstract_fk_grade_levels = []
    title_fk_grade_levels = []
    for abstract in df["Abstract Text"]:
        abstract_fk_grade_levels.append(textstat.flesch_kincaid_grade(abstract))
    for title in df["Title"]:
        title = title.lstrip("[")
        title = title.rstrip("]")
        title_fk_grade_levels.append(textstat.flesch_kincaid_grade(title))

    df["Abstract FK grade"] = abstract_fk_grade_levels
    df["Title FK grade"] = title_fk_grade_levels
    return df