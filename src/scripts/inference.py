import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


IMG_SIZE = 256 

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()
        
        self.inc = DoubleConv(in_channels, 32)       
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))   
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))  
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) 
        
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))   

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128) 
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)  
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)   
        
        # Salida
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        
        u1 = self.up1(x5)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv4(u4)
        
        return self.outc(u4)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_MODELO = os.path.join(SCRIPT_DIR, 'UNET_segmentacion_ligera.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNET(in_channels=3, out_channels=1).to(device)

try:
    model.load_state_dict(torch.load(PATH_MODELO, map_location=device))
    print("Pesos cargados correctamente.")
except RuntimeError as e:
    print("Error cargando pesos.")
    print(e)
    exit()

model.eval()
torch.backends.cudnn.benchmark = False


#PATH_IMAGEN_TEST = os.path.join(SCRIPT_DIR, 'data_set_generado', 'X_compuestas', '2_solo_simplex_LKZCAFJF.jpg') 
PATH_IMAGEN_TEST = os.path.join(SCRIPT_DIR, 'prueba_calidad.jpg') 

if not os.path.exists(PATH_IMAGEN_TEST):
    print(f"⚠️ Advertencia: No se encuentra la imagen {PATH_IMAGEN_TEST}")
    # Intenta buscar cualquier jpg en la carpeta para probar
    dir_test = os.path.dirname(PATH_IMAGEN_TEST)
    if os.path.exists(dir_test):
        files = [f for f in os.listdir(dir_test) if f.endswith('.jpg')]
        if files:
            PATH_IMAGEN_TEST = os.path.join(dir_test, files[0])
            print(f" -> Usando alternativa: {PATH_IMAGEN_TEST}")

transform_x = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

try:
    img = Image.open(PATH_IMAGEN_TEST).convert("RGB")
except Exception as e:
    print(f"Error abriendo la imagen: {e}")
    exit()

img_tensor = transform_x(img)
img_tensor = img_tensor.unsqueeze(0).to(device) 

print(f"Imagen cargada: {PATH_IMAGEN_TEST}")
print(f"Tensor shape: {img_tensor.shape}")


with torch.no_grad(): 
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > 0.5).float()
    
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Probs range:  [{probs.min():.2f}, {probs.max():.2f}]")



pred_mask = pred_mask.squeeze(0).squeeze(0).cpu()
pred_mask_scaled = pred_mask * 255.0
pred_mask_uint8 = pred_mask_scaled.type(torch.uint8)
output_image = transforms.ToPILImage()(pred_mask_uint8)

PATH_SALIDA = os.path.join(SCRIPT_DIR, 'prediccion_generada.png')
output_image.save(PATH_SALIDA)

print(f"Máscara guardada en: {PATH_SALIDA}")