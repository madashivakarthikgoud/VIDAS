<!-- Badges -->

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3109/) [![Streamlit](https://img.shields.io/badge/Streamlit-%3E%3D1.0-orange.svg)](https://streamlit.io/) [![Pandas](https://img.shields.io/badge/Pandas-%3E%3D1.2-blue.svg)](https://pandas.pydata.org/) [![NLTK](https://img.shields.io/badge/NLTK-%3E%3D3.8-green.svg)](https://www.nltk.org/) [![Plotly](https://img.shields.io/badge/Plotly-%3E%3D5.0-purple.svg)](https://plotly.com/) [![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

# VIDAS

*Video Interactive Data Analytics & Sentiment Dashboard*

[🚀 **Live Demo**](https://vidasapp.streamlit.app/) • [⭐️ **Star on GitHub**](https://github.com/madashivakarthikgoud/VIDAS)

---

## 🔍 Overview

VIDAS provides an end-to-end, interactive environment for exploring YouTube video metrics and sentiments across multiple countries. Key capabilities include:

* **Flexible Data Ingestion**: Upload your own CSV or choose from preloaded country files (CA, DE, FR, IN, JP, KR, MX, RU, US).
* **Robust Cleaning & Encoding**: Automatic encoding detection with `chardet`, type coercion, missing-value defaults, and duplicate handling.
* **Sentiment Analysis**: VADER-based scoring and categorization (Positive/Neutral/Negative) on titles and descriptions.
* **Insight Extraction**: Top hashtags, channel rankings, likes/dislikes, and comment trends.
* **Dynamic Visualizations**: Interactive Plotly charts (bar, pie, line) with customizable date ranges and top-N sliders.
* **Export Results**: Download processed data as a clean CSV for downstream tasks.

---

## 📦 Tech Stack & Dependencies

| Category           | Library / Version      |
| ------------------ | ---------------------- |
| Language           | Python 3.10+           |
| Framework          | Streamlit (≥1.0)       |
| Data Handling      | Pandas (≥1.2), chardet |
| NLP & Sentiment    | NLTK (VADER)           |
| Visualization      | Plotly Express         |
| Evaluation Metrics | scikit-learn           |

<details>
<summary><strong>requirements.txt</strong></summary>

```text
pandas
streamlit
plotly
nltk
chardet
scikit-learn
```

</details>

---

## 📂 Project Structure

```plaintext
VIDAS/
├── data/                     # Country CSVs (e.g., CAvideos.csv)
├── app.py                    # Streamlit application
├── utils.py                  # Data-loading & processing modules
├── requirements.txt
└── LICENSE                   # MIT License
```

---

## 🚀 Installation & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/madashivakarthikgoud/VIDAS.git
   cd VIDAS
   ```
2. **Create & activate virtual environment**

   ```bash
   python3.10 -m venv venv        # Requires Python 3.10+
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Launch the dashboard**

   ```bash
   streamlit run app.py
   ```

---

## 🎯 Usage Flow

1. **Sidebar**: Upload CSV or select a country.
2. **Processing**: Data cleaning and sentiment scoring.
3. **Explore**: Filter by date; adjust top-N slider for various metrics.
4. **Visualize**: Inspect interactive charts for views, sentiments, categories, and more.
5. **Export**: Download the final, cleaned dataset as CSV.

---

## 🤝 Contributing

We welcome contributions of any size!

1. ⭐️ **Star** this repo if you find it helpful.
2. 🍴 **Fork** the repository and create your feature branch.
3. 🔀 **Open a Pull Request** with a detailed description of your changes.

Help us make VIDAS even better!

---

## ⚖️ License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Crafted with passion by **Shiva Karthik***
