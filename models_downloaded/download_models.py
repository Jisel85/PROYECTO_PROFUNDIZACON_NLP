#Descarga a la máquina local el modelo, para que esté disponible de forma permanente. 

def download_models(model):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    model_name = model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Nombre de la carpeta para el modelo
    model_folder_name = model_name.replace("/", "-")

    # Directorio principal donde se guardarán el modelo y el tokenizer
    main_directory = os.getcwd()

    # Ruta completa para la carpeta del modelo
    model_directory = os.path.join(main_directory,"models_downloaded",model_folder_name)

    # Crea la carpeta si no existe
    os.makedirs(model_directory, exist_ok=True)

    # Guarda el modelo y el tokenizer en la carpeta del modelo
    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)
