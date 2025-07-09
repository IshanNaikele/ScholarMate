# backend/chains/pdf_summary_chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# For long document summarization
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Represents a piece of text with metadata

# Load environment variables from .env file at the very beginning
load_dotenv()

# Retrieve GROQ API key and ensure it's available
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize the ChatGroq model with a low temperature for consistent summaries
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key, temperature=0.1)

# --- Existing Chain (can be used for short inputs if needed) ---
# 1. Create prompt template for short summaries
prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are an expert academic summarizer. Summarize the following academic content concisely and accurately, focusing on key findings, methodologies, and conclusions. Maintain a neutral and objective tone."),
    ('user', '{input_text}')
])

# Initialize the string output parser
parser = StrOutputParser()

# Create the LangChain chain: Prompt -> Model -> Parser
# This 'chain' is suitable for inputs that fit within the LLM's context window.
chain = prompt_template | llm | parser

# --- New Function for Long Document Summarization ---
def summarize_long_document(full_text: str) -> str:
    """
    Summarizes a long document by chunking it and using LangChain's map_reduce summarization strategy.

    Args:
        full_text: The complete text of the document to be summarized.

    Returns:
        A concise summary of the entire document.
    """
    # Define the text splitter
    # chunk_size: Aim for a size well within the LLM's context window (Gemma2-9b-It is 8192 tokens).
    # 4000 characters is roughly 1000 tokens, leaving plenty of room for prompts and output.
    # chunk_overlap: Ensures context is not lost at chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, # characters
        chunk_overlap=200, # characters
        length_function=len,
        is_separator_regex=False,
    )

    # Split the document into LangChain Document objects
    # create_documents takes a list of strings and returns a list of Document objects.
    chunks = text_splitter.create_documents([full_text])

    # Define prompts for the map and reduce steps
    map_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic summarizer. Summarize the following text snippet concisely, focusing on its main points. Keep it to 2-3 sentences."),
        ("user", "{text}")
    ])

    reduce_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic summarizer. Combine the following summaries into a single, cohesive, and comprehensive summary of the entire document. Focus on key findings, methodologies, and conclusions. The final summary should be approximately 300-500 words."),
        ("user", "{text}") # {text} here will contain the concatenated summaries from the map step
    ])

    # Initialize the summarization chain with 'map_reduce' strategy
    # verbose=True helps in debugging by showing the steps of the chain in the console.
    summarization_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,      # Use custom prompt for map step
        combine_prompt=reduce_prompt_template, # Use custom prompt for combine/reduce step
        verbose=True
    )

    # Run the summarization chain on the chunks
    final_summary = summarization_chain.run(chunks)

    return final_summary

# Example Usage (for testing this file directly, won't run via FastAPI)
if __name__ == "__main__":
    # Create a long example text for demonstration
    long_document_example = """
     3.1 Encoder and Decoder Stacks
 Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
 sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position
wise fully connected feed-forward network. We employ a residual connection [11] around each of
 the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
 LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
 itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
 layers, produce outputs of dimension dmodel = 512.
 Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
 sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
 attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
 around each of the sub-layers, followed by layer normalization. We also modify the self-attention
 sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
 masking, combined with fact that the output embeddings are offset by one position, ensures that the
 predictions for position i can depend only on the known outputs at positions less than i.
 3.2 Attention
 An attention function can be described as mapping a query and a set of key-value pairs to an output,
 where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
 of the values, where the weight assigned to each value is computed by a compatibility function of the
 query with the corresponding key.
 3.2.1 Scaled Dot-Product Attention
 We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
 queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
 query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the
 values.
 In practice, we compute the attention function on a set of queries simultaneously, packed together
 into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
 the matrix of outputs as:
 Attention(Q,K,V ) = softmax(QKT
 √
 dk 
)V
 (1)
 The two most commonly used attention functions are additive attention [2], and dot-product (multi
√
 plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
 of 1
 dk 
. Additive attention computes the compatibility function using a feed-forward network with
 a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
 much faster and more space-efficient in practice, since it can be implemented using highly optimized
 matrix multiplication code.
 While for small values of dk the two mechanisms perform similarly, additive attention outperforms
 dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
 dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
 extremely small gradients 4. To counteract this effect, we scale the dot products by 1
 3.2.2 Multi-Head Attention
 √
 dk 
.
 Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
 we found it beneficial to linearly project the queries, keys and values h times with different, learned
 linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of
 queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional
 4To illustrate why the dot products get large, assume that the components of q and k are independent random
 variables with mean 0 and variance 1. Then their dot product, q · k = dk
 i=1 
qiki, has mean 0 and variance dk.
 4
output values. These are concatenated and once again projected, resulting in the final values, as
 depicted in Figure 2.
 Multi-head attention allows the model to jointly attend to information from different representation
 subspaces at different positions. With a single attention head, averaging inhibits this.
 MultiHead(Q,K,V ) = Concat(head1,...,headh)WO
 where headi = Attention(QWQ
 i ,KWK
 i ,VWV
 i )
 Where the projections are parameter matrices WQ
 i ∈Rdmodel× dk, WK
 i
 and WO ∈Rhdv×dmodel.
 ∈ Rdmodel×dk, WV
 i ∈ Rdmodel× dv
 In this work we employ h = 8 parallel attention layers, or heads. For each of these we use
 dk = dv =dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost
 is similar to that of single-head attention with full dimensionality
  3.2.3 Applications of Attention in our Model
 The Transformer uses multi-head attention in three different ways:
 • In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
 and the memory keys and values come from the output of the encoder. This allows every
 position in the decoder to attend over all positions in the input sequence. This mimics the
 typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
 [38, 2, 9].
 • The encoder contains self-attention layers. In a self-attention layer all of the keys, values
 and queries come from the same place, in this case, the output of the previous layer in the
 encoder. Each position in the encoder can attend to all positions in the previous layer of the
 encoder.
 • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
 all positions in the decoder up to and including that position. We need to prevent leftward
 information flow in the decoder to preserve the auto-regressive property. We implement this
 inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
 of the softmax which correspond to illegal connections. See Figure 2.
 3.3 Position-wise Feed-Forward Networks
 In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
 connected feed-forward network, which is applied to each position separately and identically. This
 consists of two linear transformations with a ReLU activation in between.
 FFN(x) = max(0,xW1 +b1)W2 +b2
 (2)
 While the linear transformations are the same across different positions, they use different parameters
 from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
 The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
 dff = 2048.
 3.4 Embeddings and Softmax
 Similarly to other sequence transduction models, we use learned embeddings to convert the input
 tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transfor
mation and softmax function to convert the decoder output to predicted next-token probabilities. In
 our model, we share the same weight matrix between the two embedding layers and the pre-softmax
 linear transformation, similar to [30]. In the embedding layers, we multiply those weights by √dmodel.
 
Table 1: Maximumpath lengths, per-layer complexity and minimum number of sequential operations
 for different layer types. n is the sequence length, d is the representation dimension, k is the kernel
 size of convolutions and r the size of the neighborhood in restricted self-attention.
  3.5 Positional Encoding
 Since our model contains no recurrence and no convolution, in order for the model to make use of the
 order of the sequence, we must inject some information about the relative or absolute position of the
 tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the
 bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
 as the embeddings, so that the two can be summed. There are many choices of positional encodings,
 learned and fixed [9].
 In this work, we use sine and cosine functions of different frequencies:
 PE(pos,2i) = sin(pos/100002i/dmodel)
 PE(pos,2i+1) = cos(pos/100002i/dmodel)
 where pos is the position and i is the dimension. That is, each dimension of the positional encoding
 corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We
 chose this function because we hypothesized it would allow the model to easily learn to attend by
 relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of
 PEpos.
 Wealso experimented with using learned positional embeddings [9] instead, and found that the two
 versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
 because it may allow the model to extrapolate to sequence lengths longer than the ones encountered
 during training.
 4 WhySelf-Attention
 In this section we compare various aspects of self-attention layers to the recurrent and convolu
tional layers commonly used for mapping one variable-length sequence of symbol representations
 (x1,..., xn) to another sequence of equal length (z1,...,zn), with xi,zi ∈ Rd, such as a hidden
 layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we
 consider three desiderata.
 One is the total computational complexity per layer. Another is the amount of computation that can
 be parallelized, as measured by the minimum number of sequential operations required.
 The third is the path length between long-range dependencies in the network. Learning long-range
 dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the
 ability to learn such dependencies is the length of the paths forward and backward signals have to
 traverse in the network. The shorter these paths between any combination of positions in the input
 and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare
 the maximum path length between any two input and output positions in networks composed of the
 different layer types.
 Asnoted in Table 1, a self-attention layer connects all positions with a constant number of sequentially
 executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of
 computational complexity, self-attention layers are faster than recurrent layers when the sequence
 6
length n is smaller than the representation dimensionality d, which is most often the case with
 sentence representations used by state-of-the-art models in machine translations, such as word-piece
 [38] and byte-pair [31] representations. To improve computational performance for tasks involving
 very long sequences, self-attention could be restricted to considering only a neighborhood of size r in
 the input sequence centered around the respective output position. This would increase the maximum
 path length to O(n/r). We plan to investigate this approach further in future work.
 Asingle convolutional layer with kernel width k < n does not connect all pairs of input and output
 positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels,
 or O(logk(n)) in the case of dilated convolutions [18], increasing the length of the longest paths
 between any two positions in the network. Convolutional layers are generally more expensive than
 recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity
 considerably, to O(k · n · d + n · d2). Even with k = n, however, the complexity of a separable
 convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,
 the approach we take in our model.
 Asside benefit, self-attention could yield more interpretable models. We inspect attention distributions
 from our models and present and discuss examples in the appendix. Not only do individual attention
 heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic
 and semantic structure of the sentences.
  5 Training
 This section describes the training regime for our models.
 5.1 Training Data and Batching
 We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million
 sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source
target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT
 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece
 vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training
 batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000
 target tokens.
 5.2 Hardware and Schedule
 We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using
 the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We
 trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the
 bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps
 (3.5 days).
 5.3 Optimizer
 Weused the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. We varied the learning
 rate over the course of training, according to the formula:
 lrate = d−0.5
 model · min(step_num−0.5,step_num · warmup_steps−1.5)
 (3)
 This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
 and decreasing it thereafter proportionally to the inverse square root of the step number. We used
 warmup_steps = 4000.
 5.4 Regularization
 Weemploy three types of regularization during training:
 7
Table2:TheTransformerachievesbetterBLEUscoresthanpreviousstate-of-the-artmodelsonthe
 English-to-GermanandEnglish-to-Frenchnewstest2014testsatafractionofthetrainingcost.
 Model BLEU TrainingCost(FLOPs)
 EN-DE EN-FR EN-DE EN-FR
 ByteNet[18] 23.75
 Deep-Att+PosUnk[39] 39.2 1.0·1020
 GNMT+RL[38] 24.6 39.92 2.3·1019 1.4·1020
 ConvS2S[9] 25.16 40.46 9.6·1018 1.5·1020
 MoE[32] 26.03 40.56 2.0·1019 1.2·1020
 Deep-Att+PosUnkEnsemble[39] 40.4 8.0·1020
 GNMT+RLEnsemble[38] 26.30 41.16 1.8·1020 1.1·1021
 ConvS2SEnsemble[9] 26.36 41.29 7.7·1019 1.2·1021
 Transformer(basemodel) 27.3 38.1 3.3·1018
 Transformer(big) 28.4 41.8 2.3·1019
 ResidualDropout Weapplydropout[33]totheoutputofeachsub-layer,beforeitisaddedtothe
 sub-layerinputandnormalized. Inaddition,weapplydropouttothesumsoftheembeddingsandthe
 positionalencodingsinboththeencoderanddecoderstacks.Forthebasemodel,weusearateof
 Pdrop=0.1.
 LabelSmoothing Duringtraining,weemployedlabelsmoothingofvalueϵls=0.1[36].This
 hurtsperplexity,asthemodellearnstobemoreunsure,butimprovesaccuracyandBLEUscore.
  6 Results
 6.1 MachineTranslation
 OntheWMT2014English-to-Germantranslationtask,thebigtransformermodel(Transformer(big)
 inTable2)outperformsthebestpreviouslyreportedmodels(includingensembles)bymorethan2.0
 BLEU,establishinganewstate-of-the-artBLEUscoreof28.4.Theconfigurationofthismodelis
 listedinthebottomlineofTable3.Trainingtook3.5dayson8P100GPUs.Evenourbasemodel
 surpassesallpreviouslypublishedmodelsandensembles,atafractionofthetrainingcostofanyof
 thecompetitivemodels.
 OntheWMT2014English-to-Frenchtranslationtask,ourbigmodelachievesaBLEUscoreof41.0,
 outperformingallofthepreviouslypublishedsinglemodels,atlessthan1/4thetrainingcostofthe
 previousstate-of-the-artmodel.TheTransformer(big)modeltrainedforEnglish-to-Frenchused
 dropoutratePdrop=0.1,insteadof0.3.
 Forthebasemodels,weusedasinglemodelobtainedbyaveragingthelast5checkpoints,which
 werewrittenat10-minuteintervals.Forthebigmodels,weaveragedthelast20checkpoints.We
 usedbeamsearchwithabeamsizeof4andlengthpenaltyα=0.6[38].Thesehyperparameters
 werechosenafterexperimentationonthedevelopmentset.Wesetthemaximumoutputlengthduring
 inferencetoinputlength+50,butterminateearlywhenpossible[38].
 Table2summarizesourresultsandcomparesourtranslationqualityandtrainingcoststoothermodel
 architecturesfromtheliterature.Weestimatethenumberoffloatingpointoperationsusedtotraina
 modelbymultiplyingthetrainingtime,thenumberofGPUsused,andanestimateofthesustained
 single-precisionfloating-pointcapacityofeachGPU5.
 6.2 ModelVariations
 ToevaluatetheimportanceofdifferentcomponentsoftheTransformer,wevariedourbasemodel
 indifferentways,measuringthechangeinperformanceonEnglish-to-Germantranslationonthe
 5Weusedvaluesof2.8,3.7,6.0and9.5TFLOPSforK80,K40,M40andP100,respectively.
 8
Table3:VariationsontheTransformerarchitecture.Unlistedvaluesareidenticaltothoseofthebase
 model.AllmetricsareontheEnglish-to-Germantranslationdevelopmentset,newstest2013.Listed
 perplexitiesareper-wordpiece,accordingtoourbyte-pairencoding,andshouldnotbecomparedto
 per-wordperplexities.
 N dmodel dff h dk dv Pdrop ϵls
 train PPL BLEU params
 steps (dev) (dev) ×106
 base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65
 (A)
 1 512 512 5.29 24.9
 4 128 128 5.00 25.5
 16 32 32 4.91 25.8
 32 16 16 5.01 25.4
 (B) 16 5.16 25.1 58
 32 5.01 25.4 60
 (C)
 2 6.11 23.7 36
 4 5.19 25.3 50
 8 4.88 25.5 80
 256 32 32 5.75 24.5 28
 1024 128 128 4.66 26.0 168
 1024 5.12 25.4 53
 4096 4.75 26.2 90
 (D)
 0.0 5.77 24.6
 0.2 4.95 25.5
 0.0 4.67 25.3
 0.2 5.47 25.7
 (E) positionalembeddinginsteadofsinusoids 4.92 25.7
 big 6 1024 4096 16 0.3 300K 4.33 26.4 213
 developmentset,newstest2013.Weusedbeamsearchasdescribedintheprevioussection,butno
 checkpointaveraging.WepresenttheseresultsinTable3.
 InTable3rows(A),wevarythenumberofattentionheadsandtheattentionkeyandvaluedimensions,
 keepingtheamountofcomputationconstant, asdescribedinSection3.2.2. Whilesingle-head
 attentionis0.9BLEUworsethanthebestsetting,qualityalsodropsoffwithtoomanyheads.
 InTable3rows(B),weobservethatreducingtheattentionkeysizedkhurtsmodelquality.This
 suggests thatdeterminingcompatibilityisnoteasyandthatamoresophisticatedcompatibility
 functionthandotproductmaybebeneficial.Wefurtherobserveinrows(C)and(D)that,asexpected,
 biggermodelsarebetter,anddropoutisveryhelpfulinavoidingover-fitting. Inrow(E)wereplaceour
 sinusoidalpositionalencodingwithlearnedpositionalembeddings[9],andobservenearlyidentical
 resultstothebasemodel.
 6.3 EnglishConstituencyParsing
 ToevaluateiftheTransformercangeneralizetoothertasksweperformedexperimentsonEnglish
 constituencyparsing.Thistaskpresentsspecificchallenges: theoutputissubjecttostrongstructural
 constraintsandissignificantlylonger thantheinput. Furthermore,RNNsequence-to-sequence
 modelshavenotbeenabletoattainstate-of-the-artresultsinsmall-dataregimes[37].
 Wetraineda4-layertransformerwithdmodel=1024ontheWallStreetJournal(WSJ)portionofthe
 PennTreebank[25],about40Ktrainingsentences.Wealsotraineditinasemi-supervisedsetting,
 usingthelargerhigh-confidenceandBerkleyParsercorporafromwithapproximately17Msentences
 [37].Weusedavocabularyof16KtokensfortheWSJonlysettingandavocabularyof32Ktokens
 forthesemi-supervisedsetting.
 Weperformedonlyasmallnumberofexperimentstoselectthedropout,bothattentionandresidual
 (section5.4),learningratesandbeamsizeontheSection22developmentset,allotherparameters
 remainedunchangedfromtheEnglish-to-Germanbasetranslationmodel. Duringinference,we
 9
Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23
 of WSJ)
 Parser
 Training
 WSJ23F1
 Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative
 Petrov et al. (2006) [29]
 Zhu et al. (2013) [40]
 Dyer et al. (2016) [8]
 WSJonly, discriminative
 WSJonly, discriminative
 WSJonly, discriminative
 88.3
 90.4
 90.4
 91.7
 Transformer (4 layers)
 WSJonly, discriminative
 91.3
 Zhu et al. (2013) [40]
 Huang & Harper (2009) [14]
 McClosky et al. (2006) [26]
 Vinyals & Kaiser el al. (2014) [37]
 semi-supervised
 semi-supervised
 semi-supervised
 semi-supervised
 91.3
 91.3
 92.1
 92.1
 Transformer (4 layers)
 semi-supervised
 92.7
 Luong et al. (2015) [23]
 Dyer et al. (2016) [8]
 multi-task
 generative
 93.0
 93.3
 increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3
 for both WSJ only and the semi-supervised setting.
 Our results in Table 4 show that despite the lack of task-specific tuning our model performs sur
prisingly well, yielding better results than all previously reported models with the exception of the
 Recurrent Neural Network Grammar [8].
 In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley
Parser [29] even when training only on the WSJ training set of 40K sentences.
    """  # Repeat to make it artificially long (e.g., 5000 words +)

    print("Starting long document summarization example...")
    try:
        summary = summarize_long_document(long_document_example)
        print("\n--- Long Document Summary ---")
        print(summary)
    except Exception as e:
        print(f"An error occurred during long document summarization: {e}")
        print("Please ensure your GROQ_API_KEY is correctly set and your LLM model is accessible.")