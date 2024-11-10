import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from datetime import datetime

# Configuration
class Config:
    RISK_LEVELS = {
        "Excellent": {"color": "#2ecc71", "score_range": (800, 1000), "icon": "🌟"},
        "Bon": {"color": "#27ae60", "score_range": (670, 799), "icon": "✅"},
        "Moyen": {"color": "#f39c12", "score_range": (580, 669), "icon": "⚠️"},
        "Risqué": {"color": "#e74c3c", "score_range": (300, 579), "icon": "❌"}
    }

class AdvancedCreditScoring:
    def __init__(self, model_path='model_score_credit.pkl'):
        """Initialize the credit scoring system with the pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Le modèle {model_path} n'a pas été trouvé!")
            
        self.reputation_mapping = {
            'Bon': 2,
            'Moyen': 1,
            'Faible': 0
        }
        
        self.tontine_mapping = {
            'Aucune participation': -1,
            'Participation occasionnelle': 1,
            'Active participation': 2
        }
        
        self.addiction_mapping = {
            'Addictif': -2,
            'Occasionnel': 1,
            'Non-addictif': 2
        }
        
        self.money_mapping = {
            'depot reguliers': 1,
            'depot irreguliers': 0,
            'Pas de depot': -1
        }

    def analyze_reputation(self, text):
        """
        Analyse avancée de la réputation avec prise en compte du contexte,
        des expressions idiomatiques et des nuances linguistiques.
        """
        if not text:
            return 'Moyen', 0.5

        blob = TextBlob(text)
        
        # Analyse approfondie du sentiment avec plus de granularité
        sentiment_score = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Dictionnaire enrichi de mots-clés et expressions avec pondération
        positive_keywords = {
            # Fiabilité financière
            'fiable': 0.3, 'ponctuel': 0.3, 'régulier': 0.25,
            'rembourse': 0.35, 'paiements à temps': 0.4,
            'économe': 0.3, 'gestion saine': 0.35,
            
            # Réputation sociale
            'respecté': 0.3, 'honnête': 0.35, 'responsable': 0.3,
            'recommandé': 0.35, 'intègre': 0.35, 'digne de confiance': 0.4,
            'bonne réputation': 0.35, 'apprécié': 0.25,
            
            # Comportement professionnel
            'travailleur': 0.3, 'stable': 0.3, 'sérieux': 0.35,
            'entreprenant': 0.25, 'professionnel': 0.3,
            'bon gestionnaire': 0.35, 'organisé': 0.25,
            'toujours ponctuel':0.25,
            
            # Expressions composées positives
            'toujours remboursé': 0.45, 'jamais eu de problème': 0.4,
            'très bien vu': 0.35, 'excellente réputation': 0.45,
            'membre actif': 0.3, 'personne de confiance': 0.4
        }
        
        negative_keywords = {
            # Problèmes financiers
            'retard': -0.3, 'défaut': -0.35, 'impayé': -0.4,
            'dette': -0.35, 'découvert': -0.3, 'surendetté': -0.45,
            'difficultés financières': -0.4,
            
            # Réputation négative
            'méfiance': -0.35, 'douteux': -0.4, 'mauvaise réputation': -0.45,
            'peu fiable': -0.4, 'malhonnête': -0.45, 'arnaqueur': -0.5,
            'problématique': -0.35,
            
            # Comportement à risque
            'instable': -0.35, 'irresponsable': -0.4, 'négligent': -0.35,
            'absent': -0.3, 'irrégulier': -0.35, 'gambling': -0.45,
            
            # Expressions composées négatives
            'mauvais payeur': -0.45, 'ne rembourse pas': -0.5,
            'historique de défaut': -0.45, 'problèmes récurrents': -0.4,
            'peu recommandable': -0.4, 'souvent en difficulté': -0.35
        }
        
        # Analyse contextuelle approfondie
        text_lower = text.lower()
        sentences = blob.sentences
        
        # Scores initiaux
        keyword_score = 0
        positive_count = 0
        negative_count = 0
        context_score = 0
        
        # Analyse par phrase pour capturer le contexte
        for sentence in sentences:
            sentence_text = sentence.string.lower()
            sentence_sentiment = sentence.sentiment.polarity
            
            # Vérification des mots-clés dans le contexte de la phrase
            for word, weight in positive_keywords.items():
                if word in sentence_text:
                    # Amplification si plusieurs mots positifs dans la même phrase
                    keyword_score += weight * (1 + 0.1 * positive_count)
                    positive_count += 1
                    
                    # Bonus pour les expressions composées
                    if len(word.split()) > 1 and word in sentence_text:
                        keyword_score += 0.1
                    
            for word, weight in negative_keywords.items():
                if word in sentence_text:
                    # Impact plus fort des termes négatifs dans un contexte négatif
                    keyword_score += weight * (1 + 0.1 * negative_count)
                    negative_count += 1
                    
                    # Malus pour les expressions composées négatives
                    if len(word.split()) > 1 and word in sentence_text:
                        keyword_score -= 0.1

            # Analyse des modificateurs de contexte
            context_modifiers = {
                'mais': -0.2, 'cependant': -0.2, 'néanmoins': -0.2,
                'toutefois': -0.2, 'malgré': -0.2,
                'toujours': 0.2, 'jamais': -0.2,
                'régulièrement': 0.2, 'rarement': -0.1
            }
            
            for modifier, impact in context_modifiers.items():
                if modifier in sentence_text:
                    context_score += impact * abs(sentence_sentiment)

        # Calcul du score final avec pondération des différents facteurs
        total_mentions = positive_count + negative_count
        if total_mentions > 0:
            mention_ratio = positive_count / total_mentions
        else:
            mention_ratio = 0.5

        # Score final combiné avec tous les facteurs
        final_score = (
            sentiment_score * 0.3 +  # Sentiment général
            (1 - subjectivity) * 0.15 +  # Objectivité
            keyword_score * 0.3 +  # Impact des mots-clés
            context_score * 0.15 +  # Impact du contexte
            mention_ratio * 0.1  # Ratio positif/négatif
        )
        
        # Normalisation et ajustement final
        final_score = max(0, min(1, (final_score + 1) / 2))
        
        # Classification plus nuancée
        if final_score >= 0.75:
            return 'Bon', final_score
        elif final_score >= 0.45:
            return 'Moyen', final_score
        else:
            return 'Faible', final_score

    def calculate_credit_score(self, reputation_text, tontine_level, mobile_money_status, addiction_level):
        """Calculate credit score using the pre-trained model and sentiment analysis"""
        # Analyse de la réputation
        reputation_class, sentiment_score = self.analyze_reputation(reputation_text)
        
        # Préparation des données pour le modèle
        features = {
            'Reputation': self.reputation_mapping[reputation_class],
            'Participation_tontines': self.tontine_mapping[tontine_level],
            'Niveau_adiction': self.addiction_mapping[addiction_level],
            'depot_mobile_money': self.money_mapping[mobile_money_status]
        }
        
        # Conversion en DataFrame
        X = pd.DataFrame([features])
        
        # Prédiction du modèle
        base_score = self.model.predict_proba(X)[0][1]  # Probabilité de la classe positive
        
        # Ajustement du score en fonction du sentiment avec plus de poids sur la réputation
        adjusted_score = (base_score * 0.6 + sentiment_score * 0.4) * 1000
        
        # Détermination du niveau de risque
        if adjusted_score >= 800:
            risk_level = "Excellent"
        elif adjusted_score >= 670:
            risk_level = "Bon"
        elif adjusted_score >= 580:
            risk_level = "Moyen"
        else:
            risk_level = "Risqué"
            
        return {
            'score': int(adjusted_score),
            'risk_level': risk_level,
            'reputation_class': reputation_class,
            'sentiment_score': sentiment_score,
            'features': features,
            'factors': {
                'reputation': (features['Reputation'] + 1) / 3,  # Normalisation
                'tontine': (features['Participation_tontines'] + 1) / 3,
                'mobile_money': (features['depot_mobile_money'] + 1) / 2,
                'addiction': (features['Niveau_adiction'] + 2) / 4
            }
        }

def setup_page():
    """Configuration de la page Streamlit"""
    st.set_page_config(
        page_title="Évaluation de Crédit IA",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            height: 3rem;
            margin-top: 1rem;
        }
        .score-card {
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .score-number {
            font-size: 3rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .factor-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .sentiment-analysis {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            background-color: #f1f8ff;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def show_about():
    """Page À propos"""
    st.title("À Propos du Système d'Évaluation de Crédit")
    
    st.markdown("""
    ### 🎯 Objectif du Projet
    Cette application utilise l'intelligence artificielle pour moderniser l'évaluation 
    de crédit en Afrique en intégrant des facteurs sociaux et financiers innovants.
    
    ### 🔍 Méthodologie
    Notre système combine plusieurs approches :
    - Analyse avancée de la réputation par traitement du langage naturel
    - Évaluation des comportements financiers traditionnels (tontines)
    - Analyse des transactions mobile money
    - Évaluation des facteurs de risque comportementaux
    
    ### 💡 Innovation Technologique
    - Analyse de sentiment multi-facteurs
    - Modèle d'apprentissage automatique calibré
    - Système de pondération dynamique
    - Interface utilisateur intuitive
    
    ### 📊 Facteurs d'Évaluation
    1. **Réputation** (35%)
       - Analyse de sentiment
       - Mots-clés pondérés
       - Contexte communautaire
    
    2. **Tontines** (25%)
       - Participation
       - Régularité
       - Historique
    
    3. **Mobile Money** (25%)
       - Fréquence des dépôts
       - Stabilité
       - Volume
    
    4. **Facteurs de Risque** (15%)
       - Comportement
       - Stabilité
    """)

def main():
    setup_page()
    credit_scoring = AdvancedCreditScoring()
    
    # Menu de navigation
    menu = st.sidebar.selectbox(
        "Navigation",
        ["Évaluation de Crédit", "À Propos"]
    )
    
    if menu == "À Propos":
        show_about()
        return
    
    st.title("🎯 Système d'Évaluation de Crédit IA")
    st.markdown("""
        Évaluez la solvabilité des clients en utilisant notre système d'IA qui combine 
        analyse de sentiment et facteurs socio-financiers traditionnels.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📝 Informations du Client")
        
        with st.form("credit_evaluation_form"):
            description = st.text_area(
                "Description de la réputation communautaire",
                height=100,
                placeholder="Décrivez la réputation du client dans sa communauté..."
            )
            
            col_form1, col_form2 = st.columns(2)
            
            with col_form1:
                tontine = st.selectbox(
                    "Participation à la tontine",
                    options=["Active participation", "Participation occasionnelle", "Aucune participation"]
                )
                
                mobile_money = st.selectbox(
                    "Utilisation Mobile Money",
                    options=["depot reguliers", "depot irreguliers", "Pas de depot"]
                )
            
            with col_form2:
                addiction = st.selectbox(
                    "Comportement face aux jeux",
                    options=["Non-addictif", "Occasionnel", "Addictif"]
                )
            
            submitted = st.form_submit_button("Évaluer le Score de Crédit")
    
    with col2:
        if submitted:
            st.subheader("🔍 Résultats de l'Analyse")
            
            # Calcul du score
            results = credit_scoring.calculate_credit_score(
                description, tontine, mobile_money, addiction
            )
            
            # Affichage des résultats
            color = Config.RISK_LEVELS[results['risk_level']]['color']
            icon = Config.RISK_LEVELS[results['risk_level']]['icon']
            
            # Analyse de sentiment détaillée
            st.markdown("""
                <div class='sentiment-analysis'>
                    <h4>📊 Analyse de la Réputation</h4>
                """, unsafe_allow_html=True)
            
            col_sent1, col_sent2 = st.columns(2)
            with col_sent1:
                st.metric("Classification", results['reputation_class'])
            with col_sent2:
                st.metric("Score Sentiment", f"{results['sentiment_score']:.2f}")
            
            # Score final
            st.markdown(f"""
                <div class='score-card' style='
                    background-color: {color}22;
                    border: 2px solid {color};
                '>
                    <h3>{icon} Niveau de Risque: {results['risk_level']}</h3>
                    <div class='score-number' style='color: {color}'>
                        {results['score']}
                    </div>
                    <p>Score sur 1000 points</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Détail des facteurs
            st.markdown("### 📊 Détail des Facteurs")
            for factor, value in results['factors'].items():
                st.markdown(f"""
                    <div class='factor-box'>
                        <strong>{factor.title()}:</strong>
                        <div class="progress" style="background-color: #eee; height: 10px; border-radius: 5px;">
                            <div style="width: {value * 100}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                        </div>
                        {value * 100:.1f}%
                    </div>
                """, unsafe_allow_html=True)
            
            # Affichage des recommandations
            st.markdown("### 💡 Recommandations")
            if results['risk_level'] == "Excellent":
                st.success("""
                    - Profil idéal pour l'octroi de crédit
                    - Possibilité d'augmenter les limites de crédit
                    - Eligible aux meilleurs taux d'intérêt
                """)
            elif results['risk_level'] == "Bon":
                st.info("""
                    - Profil favorable pour l'octroi de crédit
                    - Conditions standard
                    - Surveillance régulière recommandée
                """)
            elif results['risk_level'] == "Moyen":
                st.warning("""
                    - Crédit possible avec garanties supplémentaires
                    - Montant limité recommandé
                    - Suivi rapproché nécessaire
                """)
            else:
                st.error("""
                    - Risque élevé - Crédit déconseillé
                    - Nécessité d'améliorer le profil
                    - Recommandation de participation active aux tontines
                """)
            
            # Historique de l'évaluation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if 'evaluation_history' not in st.session_state:
                st.session_state.evaluation_history = []
            
            # Sauvegarde de l'évaluation dans l'historique
            evaluation_record = {
                'timestamp': timestamp,
                'score': results['score'],
                'risk_level': results['risk_level'],
                'reputation': results['reputation_class'],
                'tontine': tontine,
                'mobile_money': mobile_money,
                'addiction': addiction
            }
            st.session_state.evaluation_history.append(evaluation_record)
            
            # Affichage de l'historique
            if len(st.session_state.evaluation_history) > 1:
                st.markdown("### 📈 Historique des Évaluations")
                history_df = pd.DataFrame(st.session_state.evaluation_history)
                st.dataframe(history_df)

def init():
    """Initialisation de l'application"""
    try:
        credit_scoring = AdvancedCreditScoring()
        return True
    except Exception as e:
        st.error(f"""
        ⚠️ Erreur d'initialisation du système : {str(e)}
        
        Veuillez vérifier que :
        1. Le fichier model.pkl est présent dans le répertoire
        2. Toutes les dépendances sont installées
        """)
        return False

if __name__ == "__main__":
    if init():
        main()