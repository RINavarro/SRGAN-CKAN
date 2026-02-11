# SRGAN-CKAN

Código de investigación para **Super-Resolución de una sola imagen (SISR)** basado en un pipeline tipo **SRGAN**, donde las convoluciones estándar se sustituyen por capas de proyección inspiradas en **Kolmogorov–Arnold Networks (KAN)**.

Este repositorio contiene **únicamente la implementación** (modelos, entrenamiento y métricas).  
El desarrollo teórico y el análisis matemático se mantienen fuera de GitHub.

---

## Motivación

Los métodos clásicos de super-resolución optimizados con pérdidas pixel-a-pixel (MSE/L1) logran altos valores de PSNR, pero generan imágenes suavizadas y con pérdida de texturas de alta frecuencia.

SRGAN introdujo el uso de:
- pérdidas perceptuales,
- entrenamiento adversarial,
- y evaluación basada en percepción humana,

permitiendo reconstrucciones visualmente más realistas.

Este proyecto explora una idea complementaria:

> **Incrementar la capacidad expresiva de los operadores locales**, sustituyendo convoluciones lineales por aproximadores funcionales basados en proyección y, potencialmente, en splines (KAN).

---

## ¿Qué implementa este repositorio?

- **Generador**:
  - Arquitectura tipo SRResNet / SRGAN
  - Bloques residuales
  - Upsampling mediante `PixelShuffle`
- **Discriminador**:
  - Red convolucional con downsampling jerárquico
  - Salida en logits (entrenamiento GAN estable)
- **Pérdida perceptual**:
  - Basada en características extraídas de VGG preentrenada
- **Convolución basada en proyección (`KAN_Conv2d`)**:
  - Implementada como `unfold (im2col) + proyector`
  - El proyector actual es una capa ligera `Conv1d (1×1)`
  - La estructura permite reemplazarlo por un proyector KAN completo (splines / factorización)
- **Entrenamiento**:
  - Fase 1: preentrenamiento SRResNet
  - Fase 2: entrenamiento adversarial SRGAN
  - Soporte para *Automatic Mixed Precision* (AMP)
- **Evaluación**:
  - PSNR calculado sobre el canal de luminancia (Y)

---

## Estructura del repositorio

SRGAN-CKAN/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── models/
│   ├── Generator.py
│   ├── Discriminator.py
│   ├── ResidualBlock.py
│   ├── VGGFeatureExtractor.py
│   └── ContentLoss.py
│
├── KAN/
│   ├── KAN_Conv2d.py
│   ├── KAN_NM.py
│   ├── KANLinearNS_FactorizedLinear.py
│   └── _KANProjector.py
│
├── training/
│   ├── trainSRResNest_SRGAN.py
│   ├── train_srresnet_amp.py
│   ├── train_srgan_amp.py
│   └── training.py
│
├── metrics/
│   └── eval_epoch_metrics.py
│
├── checkpoints/
│   └── checkpoint.py
│
├── utilities/
│   └── utilities.py
│ 
├── main.py