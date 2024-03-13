import streamlit as st

import re
import time
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
import tiktoken

import matplotlib.pyplot as plt
import seaborn as sns

import grapheme
from unicodedata import category
from numpy.linalg import LinAlgError



class TokenizerAnalyzer:
    def __init__(self):
        self.tokenizers = {}

    def add_tokenizer(self, name, model_name):
        self.tokenizers[name] = model_name

    def tokenize_text(self, tokenizer_name, text):
        start_time = time.time()
        if tokenizer_name == "gpt-4":
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            tokens = tokenizer.encode(text)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizers[tokenizer_name])
            tokens = tokenizer.tokenize(text)
        end_time = time.time()
        tokenization_time = end_time - start_time
        return tokens, tokenization_time

    
    def analyze_vocab(self, vocab_file):
        latin_count = 0
        non_latin_count = 0
        latin_total_length = 0
        non_latin_total_length = 0
        incomplete_bytes_count = 0
        
        # Regular expression to match sequences starting with '\\x'
        incomplete_bytes_regex = special_char_regex = re.compile(r"(?<!\\)(\\x|\\\\x)")

        with open(vocab_file, 'r') as f:
            for line in f:
                token = re.sub(r"^(?P<quote>['\"])(.*?)(?P=quote)$", r"\2", line)
                if not "gpt-4" in vocab_file:
                    token = re.sub("_", "", token)
                token = token.strip()
                is_latin = True
                token_length = len(token)

                # Check for special character sequence at the beginning of the token
                if incomplete_bytes_regex.match(token):
                    incomplete_bytes_count += 1
                    continue  # Skip further processing for this token

                for char in token:
                    char_category = category(char)
                    if char_category != "Ll" and char_category != "Lu":  # Check for non-Latin characters
                        is_latin = False
                        break  # Exit the inner loop if a Latin character is found

                # Process token based on its category
                if is_latin:
                    latin_count += 1
                    latin_total_length += token_length
                else:
                    non_latin_count += 1
                    non_latin_total_length += token_length

        # non_latin_count += incomplete_hex_count
        #average length doe not make sense because there are tokens like: /****************************************************************
        # non_latin_count also includes cases like .WaitFor
        return {
            "latin": latin_count,
            "non_latin": non_latin_count,
            "incomplete_bytes": incomplete_bytes_count
        }
    
    
    def visualize_tokens(self, text, tokenizer):

        if tokenizer =="gpt-4":
            tokenizer = tiktoken.encoding_for_model(tokenizer)
            token_ids = tokenizer.encode(text)
            graphemes = list(grapheme.graphemes(text))
            # token_ids, str_tokens = [], []
            # for grapheme_ in graphemes:
                
            #     token_id = tokenizer.encode(grapheme_)
            #     str_tokens.append(tokenizer.decode(token_id))
            #     token_ids.append(token_id)
            str_tokens = []
            for token in token_ids:
                str_tokens.append(tokenizer.decode([token], errors="backslashreplace"))
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            tokens = tokenizer.tokenize(text)
            str_tokens = []
            for token in tokens:
                str_tokens.append(tokenizer.convert_tokens_to_string([token]))
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

        colors = ['#ffdab9', '#e6ee9c', '#9cddc8', '#bcaaa4', '#c5b0d5']
        
        html = ""
        for i, token in enumerate(str_tokens):
            color = colors[i % len(colors)]
            html += f'<mark title="{token}" style="background-color: {color};">{token}</mark>'

        st.write("Token IDs:", token_ids)
        st.write(html, unsafe_allow_html=True)

    def plot_vocab_counts(self, vocab_count_dict):
        
        outer_keys = list(vocab_count_dict.keys())
        inner_keys = list(vocab_count_dict[outer_keys[0]].keys())
        values = [[vocab[key] for key in inner_keys] for vocab in vocab_count_dict.values()]

        x = outer_keys
        num_groups = len(x)
        pastel_palette = sns.color_palette("pastel", num_groups)

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.8 / num_groups
        x_pos = [i + (1 - 0.8) / 2 for i in range(num_groups)]
        for i, y_values in enumerate(values):
            x_val = [x_pos[j] + bar_width * i for j in range(num_groups)]
            ax.bar(x_val, y_values, width=bar_width, label=x[i], color=pastel_palette[i])

            for j, value in enumerate(y_values):
                ax.annotate(str(value), xy=(x_val[j], value), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom')

        ax.set_ylabel('Count')
        ax.set_title('Vocabulary Counts')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(inner_keys, rotation=45, ha='right')
        ax.legend(title='Vocabularies', loc='upper right')

        st.pyplot(fig)

    def draw_plots(self, df, tokenizer, selected_languages):

        pastel_palette = sns.color_palette("pastel")
            
        df_selected = df[df['language'].isin(selected_languages)]
                
        plot_titles = [f"Time taken to tokenize across languages by {tokenizer}", f"Token Distribution across languages for {tokenizer}",  f"Replacement Tokens distribution across languages for {tokenizer}"]
            
        df_columns = [f"{tokenizer}_Time", f"{tokenizer}_TokensCount",  f"{tokenizer}_ReplTokensCount"]
             
        for i, column in enumerate(df_columns):
            plt.figure(figsize=(10, 6))
            try:
                sns.histplot(data=df_selected, x=column, hue="language", palette=pastel_palette, kde=True, element="step", stat="density")
                if df_selected[column].nunique() > 1 and not df_selected[column].isnull().all():
                    # Calculate mean and median
                    try:
                        mean_value = df_selected[column].mean()
                        median_value = df_selected[column].median()
                        
                        # Add vertical lines for mean and median
                        plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
                        plt.axvline(median_value, color='blue', linestyle='--', label=f'Median: {median_value:.2f}')
                        
                        # Add legend with only mean and median
                        plt.legend()
                    except LinAlgError:
                        st.warning("Singular matrix encountered. Skipping mean and median calculation.")
                
                
                plt.title(plot_titles[i])
                plt.xlabel(column.split("_")[1])
                plt.ylabel("Density")
                plt.xticks(rotation=45)
                st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"Can't Draw plot for {column}. Singular matrix encountered. Statistical measures cannot be calculated.")
        

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_selected, x="GraphemesCount", y=f"{tokenizer}_TokensCount", hue="language", palette=pastel_palette)
        plt.title(f"Graphemes vs. Token Counts across languages for {tokenizer}")
        plt.xlabel("Graphemes Count")
        plt.ylabel("Token Count")
        plt.tight_layout()
        st.pyplot(plt.gcf())

        

        
        
def playground_tab(analyzer):
    st.title("Tokenization Visualizer for Language Models")
    st.markdown("""
                You can use this playgorund to visualize tokens generated by the tokenizers used by popular language models.
        """)
    
    tokenizer_name = st.selectbox("Choose a Tokenizer", options=list(analyzer.tokenizers.keys()))
    
    text_input = st.text_area("Enter text below to visualize tokens:", height=300)
    
    if st.button("Tokenize"):
        if text_input.strip():
            analyzer.visualize_tokens(text_input, analyzer.tokenizers[tokenizer_name])
        else:
            st.error("Please enter some text.")

def analysis_tab(analyzer):
    
    st.title("Tokenizer Performance Analysis for Language Models")
    st.markdown("""
                You can use this visualizer to understand how tokenizers work across several languages. The default configuration shows results for English, French, Spanish, Hindi, Nepali.
        """)

    dataset_df = pd.read_csv("data/aya_dataset_features.csv")

    available_tokenizers = list(analyzer.tokenizers.keys())
    default_tokenizer = available_tokenizers[0]  # Change this as per your requirement
    selected_tokenizer = st.sidebar.selectbox("Select Tokenizer", options=available_tokenizers, index=available_tokenizers.index(default_tokenizer))

    languages = dataset_df["language"].unique()
    default_languages = ["English", "French", "Spanish", "Hindi", "Nepali (individual language)"]
    selected_languages = st.sidebar.multiselect("Select Languages", languages, default=default_languages)


    analyzer.draw_plots(dataset_df, selected_tokenizer, selected_languages)

# Time, Memory --> across languages across tokenizers
# replacement tokens count - across languages across tokenizers
# token distribution  - across languages across tokenizers
# graphemes v/s byte counts across languages
# graphemes v/s token counts across languages

    #Vocab counts visualization
    st.subheader("Latin v/s Non-Latin Entries in Vocab")
    st.markdown("""
                GPT-4 **cl100k_base.tiktoken** vocab contains: 
                - 70,988 entries containing only Latin characters
                - 29,268 entries containing at least one non-Latin character 
                - 803 entries with partial byte sequences
        """)
    vocab_path = ["vocab/gpt-4-vocab.txt", "vocab/nllb-vocab.txt", "vocab/roberta-vocab.txt"]
    vocab_count_dicts = {}
    for vocab in vocab_path:
        vocab_name = vocab.split("/")[-1].split(".")[0]
        vocab_count_dict = analyzer.analyze_vocab(vocab)
        vocab_count_dicts[vocab_name] = vocab_count_dict
    analyzer.plot_vocab_counts(vocab_count_dicts) 

def main():
    
    huggingface_tokenizers ={
                 "XLM-RoBERTa": "FacebookAI/xlm-roberta-base",
                 "nllb-200-distilled-600M": "facebook/nllb-200-distilled-600M",
    }
    openai_tokenizers = {
        'gpt-4': 'gpt-4',

    }

    st.sidebar.header("Welcome to Tokenization Playground")

    tabs = ['Playground', 'Analysis']
    selected_tab = selected_tab = st.sidebar.selectbox('Select from options below:', tabs)

    st.sidebar.markdown("""
                This App was created as a part of the project: "Beyond the ABCs: Exploring the nuances of tokenization in diverse languages.
                        """)   
    
    analyzer = TokenizerAnalyzer()

    for tokenizer, src in huggingface_tokenizers.items():
        analyzer.add_tokenizer(tokenizer, src)
    
    for tokenizer, _ in openai_tokenizers.items():
        analyzer.add_tokenizer(tokenizer, tokenizer)

    if selected_tab == 'Playground':
        playground_tab(analyzer)
    elif selected_tab == 'Analysis':
        analysis_tab(analyzer)
    
    
if __name__ == "__main__":
    main()

