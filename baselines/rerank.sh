for _lambda in 0.91
do
    for retriever in "bm25" "nv" "contriever"
    do
        echo "Running reranking with lambda=${_lambda} and retriever=${retriever}"
        PYTHONPATH=. python reranking/reranking.py --corpus wiki \
                                                --retriever ${retriever} \
                                                --_lambda ${_lambda} 
    done
done