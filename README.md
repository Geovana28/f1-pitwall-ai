# 🏎️ F1 Global Pit-Wall Telemetry, Car Setup & Race Strategy AI (24 Circuits • 10 Teams)

[![Live Interactive Pit-Wall](https://img.shields.io/badge/Live_Demo-GitHub_Pages-00d2be?style=for-the-badge&logo=googlechrome&logoColor=black)](https://geovana28.github.io/f1-kimi-telemetry/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Tests-9_Passed-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/Geovana28/f1-kimi-telemetry)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Plataforma Global de Engenharia de Corrida, Setup Aerodinâmico/Mecânico das 10 Equipes, Física Barométrica de Altitude e Otimização de Estratégia de Pit Stop para todos os 24 Circuitos do Calendário Mundial da Fórmula 1.**

---

## 🌍 Funcionalidades Implementadas (Sprint 1 + Sprint 2)

1. **Catálogo Mundial dos 24 Circuitos (`data/circuits_catalog.json`) + Detecção Automática do Próximo GP (`get_next_upcoming_gp`)**:
   - Cobertura completa das 24 etapas do campeonato mundial (Bahrain, Jeddah, Melbourne, Suzuka, Shanghai, Miami, Imola, Mônaco, Barcelona, Montreal, Spielberg, Silverstone, Spa-Francorchamps, Hungaroring, Zandvoort, Monza, Baku, Singapura, Austin, Cidade do México, **Interlagos / São Paulo**, Las Vegas, Qatar e Abu Dhabi).
   - Cada pista possui coordenadas GPS `(lat, lon)` integradas à API meteorológica **`Open-Meteo`** ao vivo no navegador, altitude barométrica real ($h$), alocação real de compostos Pirelli (`C1` a `C5`), fator de abrasividade do asfalto, perda de tempo no Pit Lane ($t_{\text{pit\_loss}}$) e probabilidade histórica de *Safety Car*.
2. **Catálogo das 10 Equipes & 20 Pilotos (`data/teams_catalog.json`) + Simulador Físico de Setup (`evaluate_car_setup_physics`)**:
   - Escolha qualquer uma das **10 equipes do grid** (**McLaren MCL39**, **Ferrari SF-25**, **Red Bull RB21**, **Mercedes W16**, **Aston Martin AMR25**, **Williams FW47**, **Racing Bulls VCARB 02**, **Haas VF-25**, **Alpine A525** e **Sauber/Audi C45**) e qualquer um dos 20 pilotos.
   - Ajuste o **Ângulo de Asa (*Downforce Level* 1–50)**, **Altura do Assoalho (*Ride Height* em mm)** e **Modo de Bateria (ERS Deployment)** para observar o *trade-off* físico imediato entre velocidade máxima de reta com DRS, velocidade de contorno de curva (*apex speed*) e escorregamento térmico dos pneus.
3. **Otimizador de Janela de Pit Stop e Ganho de *Undercut* (`find_optimal_one_stop_window`)**:
   - Simula volta a volta o efeito cruzado da queima de combustível com o *cliff* térmico quadrático dos compostos (`SOFT`, `MEDIUM`, `HARD`) e recomenda a volta exata de parada.

---

## 📐 Fundamentação Matemática & Física (`src/strategy_engine.py`)

### 1. Densidade Barométrica Internacional (ISA) e Redução Aerodinâmica por Altitude

$$\rho(h, T) = \rho_0 \left( 1 - \frac{L \cdot h}{T_0} \right)^{\frac{g}{R \cdot L}} \cdot \frac{T_0}{T}$$

Como as forças de arrasto e *downforce* escalam linearmente com a densidade do fluido ($F = \frac{1}{2} \rho v^2 C_L A$), a plataforma recalcula automaticamente a perda aerodinâmica de cada autódromo — ex.: **Cidade do México** ($2.240\text{ m} \implies -23{,}8\%$ ar), **Interlagos** ($765\text{ m} \implies -8{,}5\%$ ar) e **Spa-Francorchamps** ($410\text{ m} \implies -4{,}8\%$ ar).

### 2. Física de Setup do Carro (Asa, Altura do Assoalho & ERS) e Ritmo de Volta

$$T(\ell, c, a) = T_{\text{base, pista}} + \Delta_{\text{equipe, setup}} + \Delta_c + \gamma_{\text{fuel}}(L_{\text{total}} - \ell) + \theta(T_{\text{track}}, \text{setup}) \cdot \left[ \alpha_c \cdot a + \beta_c \cdot \max(0, \, a - a_{\text{cliff}, c})^2 \right]$$

Onde o desvio em relação ao ângulo ótimo de asa da pista ($\Delta w = w - w_{\text{opt}}$) afeta simultaneamente a velocidade final de reta com DRS, o tempo de contorno de curva e o fator de escorregamento térmico do pneu.

---

## 🗺️ Roadmap de Evolução (Próximas Sprints)

- **[Roadmap — Sprint 3]** Simulador estocástico de corrida por Monte Carlo ($N = 1.000$ corridas) sorteando janelas de *Safety Car* / *VSC* e *crossover* de pneu de chuva (*Intermediate* / *Wet*).
- **[Roadmap — Sprint 4]** Pipeline de ingestão de telemetria real de 240Hz via biblioteca `FastF1` / `OpenF1` + Agente Engenheiro de Box por voz/chat utilizando *Gemini Tool Calling*.

---

## 🚀 Como Executar

### 1. Plataforma Web Interativa (Zero Instalação)
Abra `index.html` diretamente no navegador ou acesse o [**Live Demo no GitHub Pages**](https://geovana28.github.io/f1-kimi-telemetry/).

### 2. Executar a Suíte Automatizada de Testes (`pytest`)
```bash
pytest tests/ -v
```

---

## 📄 Licença
Distribuído sob a licença MIT (`LICENSE`). Desenvolvido por **Geovana Morais** — [Portfólio](https://geovana28.github.io).
