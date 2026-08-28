import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

class RecSysCBF:
    def __init__(self, top_features=5000):
        self.top_features = top_features
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=5,
            max_features=self.top_features,
            stop_words='english'
        )
        self.df = None
        self.matrix = None

    def _build_item_string(self, row):
        title_str = (str(row['title']) + " ") * 2 
        skills_str = str(row['skills_desc'])
        level_str = str(row['formatted_experience_level']).replace(" ", "")
        return f"{title_str} {skills_str} {level_str}".lower()

    def fit(self, df):
        """
        Treina o vetorizador TF-IDF e guarda a matriz.
        """
        self.df = df.copy()
        
        # Preenchimento de Nulos
        self.df['title'] = self.df['title'].fillna('')
        self.df['skills_desc'] = self.df['skills_desc'].fillna('')
        self.df['formatted_experience_level'] = self.df['formatted_experience_level'].fillna('')
        
        # Merge company_name safely if exists, otherwise put empty
        if 'company_name' not in self.df.columns:
            self.df['company_name'] = 'N/A'
            
        # Feature Engineering (Item String)
        self.df['item_string'] = self.df.apply(self._build_item_string, axis=1)
        
        # Vetorização
        self.matrix = self.tfidf.fit_transform(self.df['item_string'])
        
        # CTR
        if 'applies' in self.df.columns and 'views' in self.df.columns:
            self.df['applies'] = self.df['applies'].fillna(0)
            self.df['views'] = self.df['views'].fillna(1)
            self.df['ctr'] = (self.df['applies'] / self.df['views']).clip(upper=1.0)
        else:
            self.df['ctr'] = 0.0
        
        # Remoto Flag
        if 'remote_allowed' in self.df.columns:
            self.df['is_remote'] = self.df['remote_allowed'].fillna(0).astype(int)
        else:
            self.df['is_remote'] = 0
            
        # Manter apenas as colunas que importam na visualizacao (para nao pesar a RAM)
        
        return self

    def build_user_profile(self, positive_indices, negative_indices, custom_text="", alpha=1.0, beta=0.5, gamma=1.0):
        """
        Constrói o vetor numérico do usuário
        """
        user_vector = np.zeros((1, self.matrix.shape[1]))
        
        if positive_indices:
            positive_vectors = self.matrix[positive_indices]
            user_vector += alpha * np.asarray(positive_vectors.sum(axis=0))
            
        if negative_indices:
            negative_vectors = self.matrix[negative_indices]
            user_vector -= beta * np.asarray(negative_vectors.sum(axis=0))

        if custom_text and custom_text.strip():
            text_vector = self.tfidf.transform([custom_text.lower()])
            user_vector += gamma * np.asarray(text_vector.todense())
            
        if np.linalg.norm(user_vector) > 0:
            user_vector = user_vector / np.linalg.norm(user_vector)
            
        return user_vector

    def recommend(self, user_profile, top_n=10, ctr_weight=0.2, remote_only=False):
        """
        Retorna recomendações
        """
        cosine_sim = cosine_similarity(user_profile, self.matrix).flatten()
        
        results = self.df[['job_id', 'title', 'company_name', 'formatted_experience_level', 'is_remote', 'ctr']].copy()
        results['similarity'] = cosine_sim
        
        # Bônus de CTR
        results['final_score'] = results['similarity'] * (1.0 + (ctr_weight * results['ctr']))
        
        if remote_only:
            results = results[results['is_remote'] == 1]
            
        # Filtra fora score de similaridade 0
        results = results[results['similarity'] > 0]
            
        recommended = results.sort_values(by='final_score', ascending=False)
        return recommended.head(top_n)
