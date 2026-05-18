import time

class NetworkProfiler:
    def __init__(self):
        self.latencies = []
        self.total_data_mb = 0.0

    def record_transmission(self, start_time: float, end_time: float, bytes_sent: int):
        latency = end_time - start_time
        self.latencies.append(latency)
        self.total_data_mb += bytes_sent / (1024 * 1024)

    def print_summary(self):
        if not self.latencies:
            print("No data recorded.")
            return
            
        avg_latency = sum(self.latencies) / len(self.latencies)
        total_time = sum(self.latencies)
        throughput_mbps = self.total_data_mb / total_time

        print("\n--- Telemetry Dashboard ---")
        print(f"Average Latency:  {avg_latency:.4f} seconds per tensor")
        print(f"Total Data Sent:  {self.total_data_mb:.2f} MB")
        print(f"Effective Bandwidth: {throughput_mbps:.2f} MB/s")