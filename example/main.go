package main

import (
	"flag"
	"fmt"
	"unsafe"

	"github.com/kou-m/gomenoh"
)

const (
	conv1_1InName = "140326425860192"
	fc6OutName = "140326200777584"
	softmaxOutName = "140326200803680"

	batchSize = 1
	channelNum = 3
	height = 224
	width = 224
)

var (
	onnxModelPath = flag.String("m", "data/VGG16.onnx", "onnx model path")
)

func main() {
	fmt.Println("vgg 16 example")

	flag.Parse()

	modelData, err := gomenoh.MakeModelDataFromOnnx(*onnxModelPath)
	if err != nil {
		panic(err)
	}

	vptBuilder, err := gomenoh.MakeVariableProfileTableBuilder()
	if err != nil {
		panic(err)
	}
	defer vptBuilder.Delete()

	if err := vptBuilder.AddInputProfile(conv1_1InName, gomenoh.Float, []int32{batchSize, channelNum, width, height}); err != nil {
		panic(err)
	}
	if err := vptBuilder.AddOutputProfile(fc6OutName, gomenoh.Float); err != nil {
		panic(err)
	}
	if err := vptBuilder.AddOutputProfile(softmaxOutName, gomenoh.Float); err != nil {
		panic(err)
	}

	vpt, err := vptBuilder.BuildVariableProfileTable(*modelData)
	if err != nil {
		panic(err)
	}

	modelBuilder, err := gomenoh.MakeModelBuilder(*vpt)
	if err != nil {
		panic(err)
	}
	defer modelBuilder.Delete()

	inputBuff := make([]float32, batchSize * channelNum * width * height)
	for i := 0; i < len(inputBuff); i++ {
		inputBuff[i] = 0.5
	}
	modelBuilder.AttachExternalBuffer(conv1_1InName, unsafe.Pointer(&inputBuff[0]))

	vp, err := vpt.GetVariableProfile(fc6OutName)
	if err != nil {
		panic(err)
	}
	fc6OutDataSize := 1
	for _, d := range vp.Dims {
		fc6OutDataSize *= int(d)
	}
	fc6OutData := make([]float32, fc6OutDataSize)
	modelBuilder.AttachExternalBuffer(fc6OutName, unsafe.Pointer(&fc6OutData[0]))

	model, err := modelBuilder.BuildModel(*modelData, "mkldnn", "")
	if err != nil {
		panic(err)
	}

	softmaxOutputVar, err := model.GetVariable(softmaxOutName)
	if err != nil {
		panic(err)
	}

	softmaxOutputBuff := (*[1<<30]float32)(softmaxOutputVar.BufferHandle)[:int(softmaxOutputVar.Dims[0] * softmaxOutputVar.Dims[1])]
	
	if err := model.Run(); err != nil {
		panic(err)
	}

	fmt.Printf("fc6\n%v\n", fc6OutData[:10])
	fmt.Printf("softmax\n%v\n", softmaxOutputBuff)
}
