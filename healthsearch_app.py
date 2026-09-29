"""
HealthSearch — Motor de Busca Híbrido com BM25 e Busca Semântica Vetorial
Disciplina: Tendências em Ciência da Computação — UNIPÊ
Aluno: Ricardo Silva Flores

Execução:
    pip install -r requirements.txt
    streamlit run healthsearch_app.py

O aplicativo tenta utilizar embeddings reais com o modelo multilíngue
paraphrase-multilingual-MiniLM-L12-v2. Caso o pacote, o modelo ou a conexão
não estejam disponíveis, utiliza uma simulação vetorial por conceitos médicos,
documentada na interface e no relatório.
"""

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    SENTENCE_TRANSFORMERS_DISPONIVEL = True
except ImportError:
    SENTENCE_TRANSFORMERS_DISPONIVEL = False


st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")

K_RRF = 60

DOCUMENTOS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": (
            "Pacientes com dor precordial aguda e suspeita de síndrome coronariana "
            "devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários "
            "reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": (
            "A crise hipertensiva severa requer administração de anti-hipertensivos "
            "venosos e monitoramento contínuo da pressão arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado com "
            "trombolíticos venosos em até quatro horas e meia do início dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": (
            "Parada cardiorrespiratória em adultos exige compressões torácicas "
            "contínuas de alta qualidade e desfibrilação precoce no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, "
            "recomenda-se a monitorização cardíaca contínua por telemetria."
        ),
    },
]

STOPWORDS = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "do", "dos",
    "e", "ela", "ele", "em", "entre", "esta", "este", "foi", "ha", "mais", "mas",
    "na", "nas", "no", "nos", "o", "os", "ou", "para", "pela", "pelo", "por", "que",
    "se", "sem", "sua", "sao", "um", "uma", "uns", "umas",
}


def remover_acentos(texto):
    """Converte caracteres acentuados para suas formas sem acento."""
    decomposicao = unicodedata.normalize("NFD", texto)
    return "".join(char for char in decomposicao if unicodedata.category(char) != "Mn")


def normalizar_texto(texto):
    """Normaliza caixa, acentos, pontuação e espaços consecutivos."""
    texto = remover_acentos(texto.lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto):
    """Gera tokens limpos e remove palavras funcionais frequentes."""
    return [token for token in normalizar_texto(texto).split() if token not in STOPWORDS]


def texto_indexado(documento):
    """Indexa título e conteúdo para preservar contexto e termos do protocolo."""
    return f"{documento['titulo']}. {documento['conteudo']}"


def calcular_bm25(consulta, k1, b):
    """Recria o índice BM25 com os parâmetros atuais e retorna scores por documento."""
    corpus_tokenizado = [tokenizar(texto_indexado(doc)) for doc in DOCUMENTOS]
    indice = BM25Okapi(corpus_tokenizado, k1=k1, b=b)
    scores = indice.get_scores(tokenizar(consulta))
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


CONCEITOS_MEDICOS = {
    "sindrome_coronariana": [
        "infarto", "ataque cardiaco", "miocardio", "coronariana", "isquemia miocardica",
        "precordial", "dor no peito", "angina",
    ],
    "eletrocardiograma": [
        "ecg", "eletrocardiograma", "12d", "telemetria", "monitorizacao cardiaca",
        "exame cardiaco",
    ],
    "antiagregacao": [
        "aas", "acido acetilsalicilico", "aspirina", "antiagregante", "plaquetario",
    ],
    "hipertensao": ["hipertens", "pressao arterial", "pressao alta"],
    "avc": ["avc", "acidente vascular", "derrame", "cerebral"],
    "trombolise": ["trombolitico", "trombolise", "alteplase"],
    "ressuscitacao": [
        "parada cardiaca", "parada cardio", "parada cardiorrespiratoria", "rcr", "pcr",
        "reanimacao", "desfibrilacao", "compressoes toracicas", "codigo azul",
    ],
    "terapia_intensiva": ["uti", "terapia intensiva", "monitoramento", "monitorizacao"],
    "arritmia": ["arritmia", "fibrilacao", "taquicardia", "palpitacao"],
    "urgencia": ["emergencia", "urgencia", "imediato", "minutos", "horas", "mortalidade"],
}


def embedding_simulado(texto):
    """Cria vetor de conceitos médicos para o modo de contingência offline."""
    texto_limpo = f" {normalizar_texto(texto)} "
    vetor = np.zeros(len(CONCEITOS_MEDICOS), dtype=float)
    for indice, expressoes in enumerate(CONCEITOS_MEDICOS.values()):
        vetor[indice] = sum(expressao in texto_limpo for expressao in expressoes)
    return vetor


@st.cache_resource(show_spinner=False)
def carregar_modelo_semantico():
    """Carrega o modelo real; retorna None caso não possa ser utilizado."""
    if not SENTENCE_TRANSFORMERS_DISPONIVEL:
        return None
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def gerar_vetores(textos, modelo):
    """Codifica textos pelo modelo real ou pela simulação vetorial documentada."""
    if modelo is not None:
        return np.asarray(modelo.encode(textos, show_progress_bar=False), dtype=float)
    return np.asarray([embedding_simulado(texto) for texto in textos], dtype=float)


@st.cache_resource(show_spinner=False)
def vetores_do_corpus(_modelo):
    return gerar_vetores([texto_indexado(doc) for doc in DOCUMENTOS], _modelo)


def calcular_cosseno(vetor_consulta, matriz_documentos):
    """Calcula similaridade de cosseno, protegendo vetores de norma zero."""
    produto = matriz_documentos @ vetor_consulta
    denominador = np.linalg.norm(matriz_documentos, axis=1) * np.linalg.norm(vetor_consulta)
    return np.divide(produto, denominador, out=np.zeros_like(produto, dtype=float), where=denominador != 0)


def calcular_semantica(consulta, modelo):
    matriz = vetores_do_corpus(modelo)
    vetor_consulta = gerar_vetores([consulta], modelo)[0]
    return calcular_cosseno(vetor_consulta, matriz)


def converter_scores_em_ranks(scores, somente_positivos=True):
    """Converte scores em posições; score não positivo representa documento não recuperado."""
    valores = np.asarray(scores, dtype=float)
    ranks = pd.Series(valores).rank(ascending=False, method="first").to_numpy(dtype=float)
    if somente_positivos:
        ranks[valores <= 0] = np.nan
    return ranks


def calcular_rrf(rank_lexico, rank_semantico, alpha):
    """Aplica Reciprocal Rank Fusion com k fixo igual a 60."""
    parcela_lexica = alpha * np.where(
        np.isnan(rank_lexico), 0.0, 1.0 / (K_RRF + rank_lexico)
    )
    parcela_semantica = (1.0 - alpha) * np.where(
        np.isnan(rank_semantico), 0.0, 1.0 / (K_RRF + rank_semantico)
    )
    return parcela_lexica, parcela_semantica, parcela_lexica + parcela_semantica


def executar_consulta(consulta, k1, b, alpha, modelo):
    """Executa BM25, semântica e RRF, consolidando os dados para as abas."""
    score_bm25 = calcular_bm25(consulta, k1, b)
    score_semantico = calcular_semantica(consulta, modelo)
    rank_bm25 = converter_scores_em_ranks(score_bm25)
    rank_semantico = converter_scores_em_ranks(score_semantico)
    contribuicao_bm25, contribuicao_semantica, score_rrf = calcular_rrf(
        rank_bm25, rank_semantico, alpha
    )
    rank_rrf = converter_scores_em_ranks(score_rrf, somente_positivos=False)

    return pd.DataFrame(
        {
            "ID": [doc["id"] for doc in DOCUMENTOS],
            "Título": [doc["titulo"] for doc in DOCUMENTOS],
            "Trecho clínico": [doc["conteudo"] for doc in DOCUMENTOS],
            "Score BM25": score_bm25,
            "Rank BM25": pd.array(rank_bm25, dtype="Int64"),
            "Score semântico": score_semantico,
            "Rank semântico": pd.array(rank_semantico, dtype="Int64"),
            "Contribuição BM25": contribuicao_bm25,
            "Contribuição semântica": contribuicao_semantica,
            "Score RRF": score_rrf,
            "Rank RRF": pd.array(rank_rrf, dtype="Int64"),
        }
    )


@st.cache_resource(show_spinner=False)
def carregar_cross_encoder():
    """Carrega o Cross-Encoder do bônus apenas quando o usuário o solicita."""
    if not SENTENCE_TRANSFORMERS_DISPONIVEL:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def aplicar_reranking(consulta, tabela, cross_encoder):
    """Reordena somente os três melhores documentos do RRF pelo Cross-Encoder."""
    candidatos = tabela.sort_values("Rank RRF").head(3).copy()
    pares = [
        (consulta, texto_indexado(DOCUMENTOS[indice]))
        for indice in candidatos.index
    ]
    candidatos["Score Cross-Encoder"] = cross_encoder.predict(pares)
    candidatos["Nova posição"] = (
        candidatos["Score Cross-Encoder"].rank(ascending=False, method="first").astype(int)
    )
    candidatos["Variação de posição"] = candidatos["Rank RRF"] - candidatos["Nova posição"]
    return candidatos.sort_values("Nova posição")


def exibir_ranking(tabela, coluna_score, coluna_rank):
    """Exibe tabela estável, ordenada pelo score do método solicitado."""
    colunas = [coluna_rank, "ID", "Título", coluna_score, "Trecho clínico"]
    ordenada = tabela.sort_values(coluna_score, ascending=False, kind="stable")[colunas]
    st.dataframe(ordenada, hide_index=True, use_container_width=True)


def main():
    st.title("🩺 HealthSearch")
    st.subheader("Motor de Busca Híbrido: BM25 + Busca Semântica + RRF")
    st.caption("Tendências em Ciência da Computação — UNIPÊ | Aluno: Ricardo Silva Flores")

    modelo = carregar_modelo_semantico()

    st.sidebar.header("Parâmetros de calibração")
    k1 = st.sidebar.slider("k₁ — saturação de frequência", 0.0, 3.0, 1.2, 0.1)
    b = st.sidebar.slider("b — normalização por tamanho", 0.0, 1.0, 0.75, 0.05)
    alpha = st.sidebar.slider(
        "α — peso do BM25 no RRF",
        0.0,
        1.0,
        0.5,
        0.05,
        help="α = 1 prioriza o BM25; α = 0 prioriza a busca semântica.",
    )
    ativar_reranking = st.sidebar.checkbox("Aplicar Cross-Encoder no Top-3 (bônus)")
    st.sidebar.caption(f"Constante RRF: k = {K_RRF}")

    if modelo is None:
        st.sidebar.warning(
            "Modo de simulação vetorial ativo. O modelo de embeddings não foi carregado; "
            "os vetores representam conceitos médicos documentados no código."
        )
        modo_semantico = "Simulação vetorial por conceitos médicos"
    else:
        st.sidebar.success("Embeddings reais ativos: paraphrase-multilingual-MiniLM-L12-v2")
        modo_semantico = "Embeddings reais multilíngues"

    exemplos = [
        "ataque cardíaco",
        "CÓD-ECG-12D",
        "AAS 100mg",
        "AVC isquêmico",
        "parada cardíaca",
        "crise de pressão alta",
    ]
    coluna_consulta, coluna_exemplo = st.columns([3, 1])
    with coluna_exemplo:
        exemplo = st.selectbox("Consulta de teste", exemplos)
    with coluna_consulta:
        consulta = st.text_input("Consulta clínica", value=exemplo)

    if not consulta.strip():
        st.info("Informe uma consulta para iniciar a busca.")
        return

    inicio = time.perf_counter()
    tabela = executar_consulta(consulta, k1, b, alpha, modelo)
    tempo_ms = (time.perf_counter() - inicio) * 1000

    melhor_bm25 = "—"
    if tabela["Score BM25"].max() > 0:
        melhor_bm25 = tabela.loc[tabela["Score BM25"].idxmax(), "ID"]
    melhor_semantico = "—"
    if tabela["Score semântico"].max() > 0:
        melhor_semantico = tabela.loc[tabela["Score semântico"].idxmax(), "ID"]
    melhor_rrf = tabela.loc[tabela["Score RRF"].idxmax(), "ID"]

    metrica_1, metrica_2, metrica_3, metrica_4, metrica_5 = st.columns(5)
    metrica_1.metric("Top-1 BM25", melhor_bm25)
    metrica_2.metric("Top-1 semântico", melhor_semantico)
    metrica_3.metric("Top-1 híbrido", melhor_rrf)
    metrica_4.metric("Matches léxicos", int((tabela["Score BM25"] > 0).sum()))
    metrica_5.metric("Tempo", f"{tempo_ms:.1f} ms")

    aba_bm25, aba_semantica, aba_hibrida, aba_matriz = st.tabs(
        ["Léxico — BM25", "Semântico", "Híbrido — RRF", "Matriz comparativa"]
    )

    with aba_bm25:
        st.markdown("### Busca léxica com BM25 Okapi")
        st.write(f"Tokens da consulta após normalização: `{tokenizar(consulta)}`")
        st.caption(
            "O BM25 prioriza a ocorrência exata dos termos e permite observar o efeito de k₁ e b."
        )
        if tabela["Score BM25"].max() <= 0:
            st.warning(
                "Nenhum documento recebeu score léxico positivo. Isto ilustra a limitação "
                "de uma busca por termos exatos diante de sinônimos e variações de escrita."
            )
        exibir_ranking(tabela, "Score BM25", "Rank BM25")

    with aba_semantica:
        st.markdown("### Busca por similaridade de cosseno")
        st.caption(f"Modo atual: {modo_semantico}.")
        st.latex(r"\operatorname{cos}(q,d)=\frac{q\cdot d}{\lVert q\rVert\lVert d\rVert}")
        if tabela["Score semântico"].max() <= 0:
            st.warning("A consulta não ativou conceitos ou vetores relacionados no corpus.")
        exibir_ranking(tabela, "Score semântico", "Rank semântico")

    with aba_hibrida:
        st.markdown("### Fusão Reciprocal Rank Fusion")
        st.latex(
            r"Score_{RRF}(d)=\alpha\frac{1}{60+Rank_{BM25}(d)}+"
            r"(1-\alpha)\frac{1}{60+Rank_{Sem}(d)}"
        )
        st.caption(
            "Documento com score zero em um motor não recebe rank nem contribuição desse motor. "
            "A fusão combina posições, sem normalizar escalas de score diferentes."
        )
        exibir_ranking(tabela, "Score RRF", "Rank RRF")
        st.markdown("#### Contribuição de cada motor")
        grafico = tabela.set_index("ID")[["Contribuição BM25", "Contribuição semântica"]]
        st.bar_chart(grafico)

        if ativar_reranking:
            st.markdown("#### Re-ranking Cross-Encoder do Top-3")
            with st.spinner("Carregando o Cross-Encoder, se estiver disponível..."):
                cross_encoder = carregar_cross_encoder()
            if cross_encoder is None:
                st.error(
                    "O Cross-Encoder não pôde ser carregado. Para o bônus, instale "
                    "sentence-transformers e mantenha conexão na primeira execução."
                )
            else:
                reranqueado = aplicar_reranking(consulta, tabela, cross_encoder)
                st.dataframe(
                    reranqueado[
                        [
                            "ID", "Título", "Rank RRF", "Score RRF", "Score Cross-Encoder",
                            "Nova posição", "Variação de posição",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Variação positiva significa que o documento subiu de posição. "
                    "Como o Cross-Encoder utilizado foi treinado predominantemente em inglês, "
                    "a pontuação em português deve ser interpretada apenas como demonstração técnica."
                )

    with aba_matriz:
        st.markdown("### Comparação entre os rankings")
        matriz = tabela[
            ["ID", "Título", "Rank BM25", "Rank semântico", "Rank RRF", "Score BM25", "Score semântico", "Score RRF"]
        ].copy()
        matriz["Δ RRF × BM25"] = matriz["Rank BM25"] - matriz["Rank RRF"]
        matriz["Δ RRF × Semântico"] = matriz["Rank semântico"] - matriz["Rank RRF"]
        st.dataframe(matriz.sort_values("Rank RRF"), hide_index=True, use_container_width=True)
        st.caption(
            "Delta positivo indica que o RRF melhorou a posição do documento em relação ao método comparado. "
            "Rank vazio significa que o documento não foi recuperado por aquele motor."
        )
        st.markdown("#### Leitura recomendada dos testes")
        st.markdown(
            "- **ataque cardíaco:** evidencia a cobertura semântica para sinônimos de infarto.\n"
            "- **CÓD-ECG-12D:** evidencia a precisão da busca léxica em códigos específicos.\n"
            "- **AAS 100mg:** evidencia a aproximação entre sigla e ácido acetilsalicílico."
        )


if __name__ == "__main__":
    main()
