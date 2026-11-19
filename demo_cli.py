"""
F1 Autonomous Race Engineer - Interactive Terminal CLI
Simulates live pit-wall communication with Kimi Antonelli (Mercedes-AMG Petronas F1 Team).
Shows real-time LangGraph execution trace and MCP Tool payloads.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from agent.graph import F1RaceEngineerGraph


def print_banner():
    banner = r"""
================================================================================
🏎️  F1 AUTONOMOUS RACE ENGINEER • LANGGRAPH & MCP ARCHITECTURE
    Driver: Andrea Kimi Antonelli (#12) • Mercedes W17 • Interlagos GP
    Frameworks: LangGraph StateGraph | Model Context Protocol (v1.0) | Scikit-Learn
================================================================================
"""
    print(banner)


def run_cli():
    print_banner()
    graph = F1RaceEngineerGraph()

    scenarios = [
        "Qual é a nossa probabilidade matemática de pódio largando de P2 em Interlagos?",
        "Bono, o desgaste do Médio está alto na volta 22. Devemos parar no box agora?",
        "Qual é a leitura de telemetria completa da volta atual?"
    ]

    print("Exemplos de chamadas pelo rádio:")
    for idx, sc in enumerate(scenarios, 1):
        print(f"  [{idx}] \"{sc}\"")
    print("  [0] Digitar mensagem personalizada\n")

    while True:
        try:
            choice = input("\n🎙️ [PILOTO / RÁDIO] Escolha uma opção (ou digite 'sair'): ").strip()
            if choice.lower() in ["sair", "exit", "quit", "q"]:
                print("\n🏁 Desconectando rádio do box. Até a próxima corrida!\n")
                break

            if choice == "1":
                query = scenarios[0]
            elif choice == "2":
                query = scenarios[1]
            elif choice == "3":
                query = scenarios[2]
            elif choice == "0" or len(choice) > 5:
                query = choice if len(choice) > 5 else input("Digite a mensagem de rádio: ").strip()
            else:
                query = scenarios[0]

            print(f"\n📡 Transmitindo via rádio para o Engenheiro de Pista autônomo...")
            print(f"💬 \"{query}\"\n")
            print("─" * 70)
            print("⚙️  [LANGGRAPH TRACE] Executando nós do StateGraph:")

            state = graph.run(
                user_query=query,
                driver="Andrea Kimi Antonelli",
                circuit="Interlagos",
                lap=22,
                grid=2
            )

            for step in state.execution_trace:
                node_name = step.get("node")
                action = step.get("action")
                print(f"  ├── 🔄 [{node_name}] -> {action}")
                if "scheduled_tools" in step:
                    print(f"  │    └── Tools agendadas: {step['scheduled_tools']}")
                if "executed_tools" in step:
                    for t in step["executed_tools"]:
                        print(f"  │    └── ⚡ MCP Tool executada: {t['tool']} (Status: {'OK' if t['success'] else 'ERR'})")
                if "safety_certification" in step:
                    print(f"  │    └── 🛡️ FIA Safety Critique: {step['safety_certification']}")

            print("─" * 70)
            print("\n" + state.radio_response)
            print(f"\n🏷️  Código Tático: [{state.tactical_code}]")
            print("═" * 70)

        except KeyboardInterrupt:
            print("\nSessão encerrada.")
            break


if __name__ == "__main__":
    run_cli()
