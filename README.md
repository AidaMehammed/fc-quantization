# [FeatureCloud Quantization](https://github.com/AidaMehammed/Quantization_CNN)
### Model Compression with Quantization

The FC Quantization package provides a comprehensive solution for model compression through quantization techniques. Integrated with federated learning frameworks and built upon the capabilities of PyTorch's quantization functionalities, this package facilitates efficient distributed training suitable for various machine learning tasks.


1. Configure Quantization:
    - Set up initial quantization settings.
      ```python
        self.configure_quant(model, backend, quant_typ)
        ```

2. Train Local Model:
    - If using post-static quantization:
        - Train the model.
        - Prepare for post-static quantization.
         ```python
         prep_post_static_quant(model, train_loader, backend)
        ```

    - Else:
        - Prepare for quantization-aware training.
      ```python
      pf.prepare_qat(model,backend)        
      ```
        - Train the prepared model.


3. Reconfigure Quantization:
    - Update quantization settings for the trained model.
   ```python
        self.configure_quant(prepared_model, backend, quant_typ)
   ```

4. Send Data to Coordinator:
    - Send the prepared model data to the coordinator.