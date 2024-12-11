import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CycleGANTrainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize generators and discriminators
        self.G_style_to_content = Generator().to(self.device)
        self.G_content_to_style = Generator().to(self.device)
        self.D_style = Discriminator().to(self.device)
        self.D_content = Discriminator().to(self.device)
        
        # Loss functions
        self.adversarial_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        
        self.optimizer_G = optim.Adam(
            list(self.G_style_to_content.parameters()) + 
            list(self.G_content_to_style.parameters()),
            lr=config['learning_rate'], betas=(0.5, 0.999)
        )
        self.optimizer_D_style = optim.Adam(
            self.D_style.parameters(), 
            lr=config['learning_rate'], betas=(0.5, 0.999)
        )
        self.optimizer_D_content = optim.Adam(
            self.D_content.parameters(), 
            lr=config['learning_rate'], betas=(0.5, 0.999)
        )
    
    def train(self, dataloader, epochs):
        for epoch in range(epochs):
            for batch in dataloader:
                style_specs = batch['style'].to(self.device)
                content_specs = batch['content'].to(self.device)
                
                # Adversarial ground truths
                valid = torch.ones((style_specs.size(0), 1, 1, 1), requires_grad=False).to(self.device)
                fake = torch.zeros((style_specs.size(0), 1, 1, 1), requires_grad=False).to(self.device)
                
                # Train Generators
                self.optimizer_G.zero_grad()
                
                # Cycle Consistency Loss
                style_to_content = self.G_style_to_content(style_specs)
                content_to_style = self.G_content_to_style(content_specs)
                
                cycle_style = self.G_content_to_style(style_to_content)
                cycle_content = self.G_style_to_content(content_to_style)
                
                # Adversarial Loss
                d_style_fake = self.D_style(content_to_style)
                d_content_fake = self.D_content(style_to_content)
                
                g_loss_style = self.adversarial_loss(d_style_fake, valid)
                g_loss_content = self.adversarial_loss(d_content_fake, valid)
                
                # Cycle Consistency Loss
                cycle_loss_style = self.cycle_loss(cycle_style, style_specs)
                cycle_loss_content = self.cycle_loss(cycle_content, content_specs)
                
                # Total Generator Loss
                total_g_loss = (
                    g_loss_style + g_loss_content + 
                    10 * cycle_loss_style + 10 * cycle_loss_content
                )
                
                total_g_loss.backward()
                self.optimizer_G.step()
                
                # Train Discriminators
                # Style Discriminator
                self.optimizer_D_style.zero_grad()
                d_style_real = self.D_style(style_specs)
                d_style_fake = self.D_style(content_to_style.detach())
                
                d_style_loss_real = self.adversarial_loss(d_style_real, valid)
                d_style_loss_fake = self.adversarial_loss(d_style_fake, fake)
                d_style_loss = (d_style_loss_real + d_style_loss_fake) * 0.5
                
                d_style_loss.backward()
                self.optimizer_D_style.step()
                
                # Content Discriminator
                self.optimizer_D_content.zero_grad()
                d_content_real = self.D_content(content_specs)
                d_content_fake = self.D_content(style_to_content.detach())
                
                d_content_loss_real = self.adversarial_loss(d_content_real, valid)
                d_content_loss_fake = self.adversarial_loss(d_content_fake, fake)
                d_content_loss = (d_content_loss_real + d_content_loss_fake) * 0.5
                
                d_content_loss.backward()
                self.optimizer_D_content.step()

def train_music_style_transfer(style_path, content_path):
    # Configuration
    config = {
        'learning_rate': 0.0002,
        'batch_size': 16,
        'epochs': 100
    }
    
    # Dataset
    dataset = MusicStyleDataset(style_path, content_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # Trainer
    trainer = CycleGANTrainer(config)
    trainer.train(dataloader, config['epochs'])

# Main execution example
if __name__ == '__main__':

    cont = 'FMA_trial/content'
    sty = 'FMA_trial/style'

    train_music_style_transfer(
        style_path=sty, 
        content_path=cont
    )
