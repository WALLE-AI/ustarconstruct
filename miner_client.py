from PIL import Image
from mineru_vl_utils import MinerUClient

client = MinerUClient(
    backend="http-client",
    model_name="MinerU2.5-2509-1.2B",
    server_url="http://10.147.45.228:60002"
)

image = Image.open("wechat_2025-10-22_144349_916.png")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)