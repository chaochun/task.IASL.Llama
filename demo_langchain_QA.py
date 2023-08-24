import os

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma

import torch
from typing import Optional
import fire
from llama import Llama

def write_line(nchars=10):
    print('-'*nchars)
    return

class DocumentRetriever:
    def __init__(self, doc_encoder=None, doc_paths=None):
        self.doc_paths = doc_paths
        self.doc_encoder = doc_encoder

        self.vectorstore = None
        
        self.construct_document_store()
        return

    def load_text(self, ipath):
        data = None
        with open(ipath, "r", encoding='utf-8') as file_txt:
            data = file_txt.read()
        file_txt.close()

        assert data!=None

        return data
    
    def construct_document_store(self):
        documents = []

##      @20230823, check if all pdf or txt documents
        PDF_TYPE = False
        for file_path in self.doc_paths:
            if '.pdf' in file_path:
                PDF_TYPE = True
                break

        for file_path in self.doc_paths:
            if PDF_TYPE:
                assert '.pdf' in file_path
            else:
                assert '.txt' in file_path
##      @20230823

        text_splitter = None
##      initialize text_splitter according to the document type (different separators will be assigned)
        if PDF_TYPE:
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        else:
            text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=200)

        for file_path in self.doc_paths:
            if '.pdf' in file_path:
                print('>>LOAD-PDF: %s'%(file_path))
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            else:
                print('>>LOAD-TXT: %s'%(file_path))
                data = self.load_text(file_path)
                txtdoc = text_splitter.create_documents([data])
                documents.extend(txtdoc)

        texts = text_splitter.split_documents(documents)

        print('>>pdf-documents loaded (#-%d) !!'%(len(self.doc_paths)))
        print('>>spilitted text (#-%d) !!'%(len(texts)))

##      DEBUGGING
        # print('>>dump the spilitted texts')
        # for i, text in enumerate(texts):
        #     print('>>Text-%d'%(i+1))
        #     write_line()
        #     print(text)
        #     write_line()
        # print('>>done')
##      DEBUGGING                

        print('>>constructing local document store')
        self.vectorstore = Chroma.from_documents(texts, self.doc_encoder)
        print('>>local document store completed !')

        return

    def retrieve_best_document(self, query=None, topn=3): ## default of Chroma.similarity_search() is 4
        if query==None:
            print('WARNING: No query string and returned (from ::retrieve_best_document()) !!')
            return None

        # query = "How many patients with diabetes come from America?"
    
        print('>>searching documents for query=[%s] ...'%(query))
        docs = self.vectorstore.similarity_search(query, k=topn)

##      return top1
        # best_match = docs[0].page_content

##      return topn        
        out_relevant_docs = []
        for doc in docs:
            out_relevant_docs.append(doc.page_content)

##      DEBUGGING        
        print('>>dump the best-match candidate(s)')
        for i, doc in enumerate(out_relevant_docs):
            print('>>TOP-%d'%(i+1))
            write_line()
            print(doc)
            write_line()
        print('>>done')
##      DEBUGGING

        return out_relevant_docs

class DocumentQA:
    def __init__(self, doc_encoder=None, doc_paths=None, llama_chat_agent=None, max_gen_len=None, temperature=None, top_p=None):
        self.doc_paths = doc_paths
        self.doc_encoder = doc_encoder

        self.llama_chat_agent = llama_chat_agent
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p

        self.doc_retriver = None

        self.initializer()
        return

    def initializer(self):
        self.doc_retriver = DocumentRetriever(doc_encoder=self.doc_encoder, doc_paths=self.doc_paths)
        return
    
    def renew_llama_query(self, question=None):

        out_relevant_docs = self.doc_retriver.retrieve_best_document(query=question)

        llama_queries = []
        for doc in out_relevant_docs:
            llama_query = f"Answer the question based on the specified article provided between <PARA> and </PARA>, \n\
            If the answer is not within the designated article in <PARA> and </PARA>, please respond with \"The answer is beyond the content of the article.\"\n\n\
            <PARA>\n{doc}\n</PARA>\n\nMy question is: \n{question}\n"

            llama_queries.append(llama_query)

##      DEBUGGING
        # print('>>dump llama_query ___begin')
        # write_line()
        # print(llama_query)
        # write_line()
        # print('>>dump llama_query ___end')
##      DEBUGGING

        return llama_queries

    def run_document_qa(self, question=None):

        # question = "How many patients with diabetes come from America?"

        llama_chat_agent = self.llama_chat_agent

        contents = self.renew_llama_query(question=question)

        nbest_answers = []
        for i, content in enumerate(contents):
            print('\n')
            print('>>PROMPT: %s'%(content))

            dialogs = []
            dialog = [{"role": "user", "content": "%s"%(content)}]
            dialogs.append(dialog)

            print('>>perform ::llama_chat_agent.chat_completion() ___begin')
            results = llama_chat_agent.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            print('>>perform ::llama_chat_agent.chat_completion() ___end')
            
            dialog = dialogs[-1]
            result = results[-1]

            answer = result['generation']['content']
            answer = answer.strip()
            nbest_answers.append(answer)

        print('\n\n')
        print('LLAMA-2 CHAT-QA ANSWER REPORT')
        write_line()
        print('QUESTION:', question)
        for i, answer in enumerate(nbest_answers):
            print('ANSWER[%d]: %s'%(i+1,answer))
        print('\n\n')

        return

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0, ##@20230804, default=0.6
    top_p: float = 0.9,
    max_seq_len: int = 2048, ##@20230804, default=512
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):

    LOAD_AS_PDF_DOCUMENTS = True
    # LOAD_AS_PDF_DOCUMENTS = False

    if torch.cuda.is_available():
        print('>>CUDA GPU is available')
    else:
        print('>>RUN in CPU mode')

    if LOAD_AS_PDF_DOCUMENTS:
        ipath1 = os.path.join('data', 'cli000.pdf')
    else:
        ipath1 = os.path.join('data', 'cli000.txt')
    ipaths = [ipath1]

##  initialize HuggingFace Text Embedding
##  @20230823, to avoid the "RuntimeError: expected scalar type Float but found Half" error, load the HF embedding first
    print('>>loading HF embedding ___begin')
    embeddingHF = HuggingFaceInstructEmbeddings()
    print('>>loaded HF embedding ___done')

##  initialize the LLama-2 chat model
    print('>>loading Llama-2 chat model ___begin')
    llama_chat_agent = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print('>>loading Llama-2 chat model ___end')

##
    agentDocQA = DocumentQA(doc_encoder=embeddingHF, doc_paths=ipaths, llama_chat_agent=llama_chat_agent, \
      max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

##
    questions = [\
    "How many patients with diabetes come from America?", \
    "How many phase I trials are in the clinical programme?", \
    "Which continents do these people come from?"]

    for ques in questions:
        agentDocQA.run_document_qa(question=ques)

    print('886!')
    return


if __name__ == "__main__":
#    
    fire.Fire(main)
##
    # debug_retrival_document()
