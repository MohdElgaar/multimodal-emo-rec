# Multimodal Emotion Recognition

- This is a framework for traninig a multimodal sequence classification model end-to-end. The method can be 'full' or 'auxiliary', 'staged' or 'one-stage', 'tuning' or 'non-tuning'
by setting *hparams.setting*, *hparams.tune_prev*, and *hparams.init_stage*. 

- Data augmentation can be applied to each of audio, text, and video by randomly sampling *segment_length* samples from the time sequence in step of *step_size*. In python code it randomly selects a *start_idx*, then: `segment = audio[start_id : start_id + audio_segment * audio_step : audio_step]`. This can be done by settingg *hparams.audio_segment*, *hparams.audio_step*, and similiraly for *img* and *text*.

- Also provided is preprocessing code for feature extraction and saving to disk.

- The framework also implements a method of sub-sampling from the dataset to fit the data of all modalities on memory. And the sub-sampled dataset is updated every *hparams.update_evey* sub-epochs.
