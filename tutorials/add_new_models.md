# Executar um novo modelo de LLM

1. Caso seja necessário adicionar um novo modelo de LLM na execução do PALIMPZEST, o arquivo "abacus/lib/python3.12/site-packages/palimpzest/constants.py" deve ser alterado da seguinte forma:
    -Adicionar o enum na classe Model;
    -Criar um MODEL_CARDS do modelo;
    -Adicionar o modelo no atributo MODEL_CARDS;

2. Exemplo: Vamos supor que eu queira adicionar o modelo "meta-llama/Llama-3.1-8B-Instruct".
    - Dentro da classe Model, eu adicionaria a linha:  
        **VLLM_LLAMA_3_1_8B_INSTRUCT = "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"**

    - O MODEL_CARDS, com valores arbritrários, seria adicionado desta forma: 
        VLLM_LLAMA_3_1_8B_INSTRUCT_MODEL_CARD = {
            ##### Cost in USD #####
            "usd_per_input_token": 0.0 / 1e6,
            "usd_per_output_token": 0.0 / 1e6,
            ##### Time #####
            "seconds_per_output_token": 0.1000, # TODO: fill-in with a better estimate
            ##### Agg. Benchmark #####
            "overall": 30.0, # TODO: fill-in with a better estimate
        }

    - Por fim, o atributo MODEL_CARDS teria a seguinte linha adicionada:
         **Model.VLLM_LLAMA_3_1_8B_INSTRUCT.value : VLLM_LLAMA_3_1_8B_INSTRUCT_MODEL_CARD,**

3. Dessa forma, para que o modelo seja executado, a variável config pode receber os seguintes valores (não esquecer de verificar qual a porta onde está o servidor do VLLM):
    config = pz.QueryProcessorConfig(
        available_models=["hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"],
        api_base="http://localhost:8001/v1",
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        progress=True
    )