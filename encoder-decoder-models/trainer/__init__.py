from .language import EncoderDecoderTrainer
from .vision_language import VisionLanguageEncoderDecoderTrainer

def get_trainer(model, model_filename, enc_tokenizer, dec_tokenizer, 
                 train_loader, valid_loader, test_loader,
                 optim, scheduler, sep_token, target_mode,
                 gradient_accumulation_steps, device, epoches, logger):
    
    if 'visual' in model_filename:
        return VisionLanguageEncoderDecoderTrainer(model, model_filename, enc_tokenizer, dec_tokenizer, 
                    train_loader, valid_loader, test_loader,
                    optim, scheduler, sep_token, target_mode,
                    gradient_accumulation_steps, device, epoches, logger)
    else:
        return EncoderDecoderTrainer(model, model_filename, enc_tokenizer, dec_tokenizer, 
                    train_loader, valid_loader, test_loader,
                    optim, scheduler, sep_token, target_mode,
                    gradient_accumulation_steps, device, epoches, logger)