# 🏎️ F1 Pit-Wall Telemetry & Race Strategy AI — Mexico City GP (Kimi Antonelli #12 Edition)

[![Live Interactive Pit-Wall](https://img.shields.io/badge/Live_Demo-GitHub_Pages-00d2be?style=for-the-badge&logo=googlechrome&logoColor=black)](https://geovana28.github.io/f1-kimi-telemetry/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest_Passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/Geovana28/f1-kimi-telemetry)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Engenharia de Telemetria, Física de Altitude e Otimização de Estratégia de Pit Stop na Fórmula 1.**  
> Inspirado na vitória histórica de **Kimi Antonelli (#12)** no **Grande Prêmio da Cidade do México (Autódromo Hermanos Rodríguez — 2.240m)**.

---

## 🎯 O Problema Real do Pit-Wall no GP do México

O Autódromo Hermanos Rodríguez (71 voltas, $4{,}304\text{ km}$) apresenta dois desafios físicos e matemáticos únicos no calendário da Fórmula 1:
1. **Física Atmosférica em Grande Altitude ($h = 2.240\text{ m}$)**: O ar rarefeito reduz a densidade atmosférica ($\rho$) em **$\approx 23{,}8\%$** em relação ao nível do mar, diminuindo drasticamente tanto o arrasto aerodinâmico ($F_d$) quanto o *downforce* ($F_L$) e a capacidade de refrigeração dos freios e da unidade de potência.
2. **Trade-off Não-Linear entre Queima de Combustível e *Cliff* Térmico dos Pneus**: A cada volta o carro fica $\approx 1{,}5\text{ kg}$ mais leve (ganhando $\approx 0{,}055\text{ s/volta}$), mas o desgaste térmico do composto (Soft, Medium, Hard) sofre uma transição quadrática abrupta após a volta limite (*tire cliff*), abrindo a janela matemática para o **Undercut**.

---

## 📐 Fundamentação Matemática & Física Implementada (`src/strategy_engine.py`)

### 1. Densidade Barométrica Internacional (ISA) e Redução Aerodinâmica

$$\rho(h, T) = \rho_0 \left( 1 - \frac{L \cdot h}{T_0} \right)^{\frac{g}{R \cdot L}} \cdot \frac{T_0}{T}$$

Como as forças de arrasto e *downforce* escalam linearmente com a densidade do fluido ($F = \frac{1}{2} \rho v^2 C_L A$), a $2.240\text{ m}$ e $24^\circ\text{C}$ obtemos $\rho \approx 0{,}933\text{ kg/m}^3$ vs. $\rho_0 = 1{,}225\text{ kg/m}^3$ ao nível do mar (**$-23{,}8\%$ de pressão aerodinâmica**).

### 2. Modelo de Tempo de Volta com *Cliff* Térmico Não-Linear

$$T(\ell, c, a) = T_{\text{base}} + \Delta_c + \gamma_{\text{fuel}}(L_{\text{total}} - \ell) + \theta(T_{\text{track}}) \cdot \left[ \alpha_c \cdot a + \beta_c \cdot \max(0, \, a - a_{\text{cliff}, c})^2 \right]$$

Onde:
- $\ell \in \{1, \dots, 71\}$ é a volta da corrida e $a$ é a idade do pneu no *stint* atual;
- $\gamma_{\text{fuel}} = 0{,}055\text{ s/volta}$ modela o ganho de tempo pela redução de massa de combustível;
- $\alpha_c$ e $\beta_c$ são as taxas de degradação linear e quadrática pós-*cliff* ($a_{\text{cliff}}$) de cada composto (`SOFT`, `MEDIUM`, `HARD`).

### 3. Otimização da Janela de Pit Stop e Ganho de *Undercut*

$$\ell_{\text{pit}}^* = \arg\min_{\ell_{\text{pit}} \in [15, 55]} \left( \sum_{\ell=1}^{\ell_{\text{pit}}} T(\ell, c_1, \ell) + t_{\text{pit\_loss}} + \sum_{\ell=\ell_{\text{pit}}+1}^{L_{\text{total}}} T(\ell, c_2, \ell - \ell_{\text{pit}}) \right)$$

---

## 🚀 Como Executar

### 1. Simulador Interativo (Zero Instalação)
Abra `index.html` diretamente no navegador ou acesse o [**Live Demo no GitHub Pages**](https://geovana28.github.io/f1-kimi-telemetry/).

### 2. Executar a Suíte Automatizada de Testes (`pytest`)
```bash
pytest tests/ -v
```

---

## ⚠️ Hipóteses e Limitações Metodológicas

1. **Pista Livre (*Clean Air*)**: O otimizador determinístico calcula o ritmo em pista livre, abstraindo perda de tempo em tráfego (*dirty air* / trem de DRS) e intervenções estocásticas de *Safety Car* ou *Virtual Safety Car (VSC)*.
2. **Parâmetros Calibrados**: Os coeficientes $\alpha_c, \beta_c$ e $t_{\text{pit\_loss}} = 21{,}8\text{ s}$ representam valores médios de referência do circuito da Cidade do México.

---

## 📄 Licença
Distribuído sob a licença MIT (`LICENSE`). Desenvolvido por **Geovana Morais** — [Portfólio](https://geovana28.github.io).
