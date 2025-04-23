# SDXL on AWS Inferentia2

이 프로젝트는 **Stable Diffusion XL (SDXL)** 모델을 **AWS Inferentia2** 기반 **Amazon SageMaker** 환경에서 실행하기 위한 실험 및 배포 과정을 정리한 저장소입니다. CUDA 환경에서의 모델 실행 테스트부터 시작해, Neuron SDK를 활용한 Inferentia2 전용 컴파일 및 엔드포인트 배포, 그리고 실시간 추론까지 포함합니다.

## 📂 프로젝트 구성

```
sdxl_inf2/
├── SDXL_test_on_cuda.ipynb         # CUDA 환경에서 SDXL 모델 실행 테스트
├── Stable_Diffusion_on_INF2.ipynb  # Neuron SDK 기반 모델 컴파일 및 S3 업로드
├── invoke_sdxl_inf2.ipynb          # SageMaker 엔드포인트 호출 및 이미지 생성
├── requirements.txt                # 실행에 필요한 Python 패키지 목록
├── templates.py                    # 텍스트 프롬프트 생성 도우미
├── output_image.jpg                # 생성 이미지 샘플
├── sample_image_M_40_01.png        # 샘플 프롬프트 이미지
└── README.md
```

## 🧪 1. CUDA 환경에서 SDXL 실행

`SDXL_test_on_cuda.ipynb` 노트북에서는 CUDA 환경에서 SDXL 모델을 로드하고 텍스트 기반 이미지 생성 기능을 테스트합니다.

필요 패키지 설치:
```bash
pip install diffusers transformers accelerate
```

실행 예시:
```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to("cuda")

image = pipe("a fantasy castle at sunset").images[0]
image.save("output_image.jpg")
```

## ⚙️ 2. Inferentia2 용 모델 컴파일 및 업로드

`Stable_Diffusion_on_INF2.ipynb` 노트북에서는 Optimum Neuron을 사용하여 모델을 Neuron용으로 컴파일한 후, Amazon S3에 업로드합니다.

필요 패키지 설치:
```bash
pip install "optimum-neuron[diffusers]" sagemaker
```

## 🚀 3. SageMaker Endpoint 배포 및 추론

컴파일된 모델을 SageMaker에서 `ml.inf2.xlarge` 인스턴스로 호스팅하고, `invoke_sdxl_inf2.ipynb`를 통해 엔드포인트에 프롬프트를 전달하여 실시간 이미지를 생성합니다.

예시 요청:
```python
import boto3, json, base64
from PIL import Image
from io import BytesIO

client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    EndpointName="your-endpoint-name",
    ContentType="application/json",
    Body=json.dumps({"prompt": "a futuristic city at dawn"})
)

result = json.loads(response["Body"].read())
img_data = base64.b64decode(result["image_base64"])
Image.open(BytesIO(img_data)).show()
```

## 🔐 보안 가이드라인

이 프로젝트에서는 다음의 보안 모범 사례를 따릅니다:

- **IAM 최소 권한 원칙**: SageMaker와 S3에 접근하는 IAM 역할은 필요한 권한만 부여합니다.
- **S3 암호화 사용**: 모델 아티팩트는 SSE-KMS로 암호화되어 저장됩니다.
- **VPC 내 엔드포인트 배포**: 공용 인터넷에 노출되지 않도록 설정합니다.
- **CloudWatch 로깅 활성화**: 추론 요청 및 오류 로그를 수집합니다.

## 📚 참고 자료

- [Stable Diffusion XL 모델](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Optimum Neuron 문서](https://huggingface.co/docs/optimum-neuron/)
- [Amazon SageMaker Neuron 인스턴스 가이드](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-neuron.html)
