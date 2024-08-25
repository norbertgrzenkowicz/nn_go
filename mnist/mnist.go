package mnist

import (
	"encoding/binary"
	"errors"
	"image/color"
	"io"
	"os"
	"path"
)

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
)

const (
	TrainingImageFileName = "train-images.idx3-ubyte"
	TrainingLabelFileName = "train-labels.idx1-ubyte"
	TestImageFileName     = "t10k-images.idx3-ubyte"
	TestLabelFileName     = "t10k-labels.idx1-ubyte"
)

type Image [Width * Height]byte

const (
	Width  = 28
	Height = 28
)

type Set struct {
	Images []*Image
	Labels []int8
}

var (
	ErrFormat = errors.New("mnist: invalid format")

	ErrSize = errors.New("mnist: size mismatch")
)

type labelFileHeader struct {
	Magic     int32
	NumLabels int32
}

type imageFileHeader struct {
	Magic     int32
	NumImages int32
	Height    int32
	Width     int32
}

func readImage(r io.Reader) (*Image, error) {
	img := &Image{}
	err := binary.Read(r, binary.BigEndian, img)
	return img, err
}

func LoadImages(name string) ([]*Image, error) {
	file, err := os.Open(name)

	if err != nil {
		return nil, err
	}
	defer file.Close()

	var reader io.Reader = file
	header := imageFileHeader{}
	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != imageMagic ||
		header.Height != Height ||
		header.Width != Width {
		return nil, ErrFormat
	}

	images := make([]*Image, header.NumImages)

	for i := int32(0); i < header.NumImages; i++ {
		images[i], err = readImage(reader)

		if err != nil {
			return nil, err
		}
	}

	return images, nil
}

func LoadLabels(name string) ([]int8, error) {
	file, err := os.Open(name)

	if err != nil {
		return nil, err
	}
	defer file.Close()

	var reader io.Reader = file
	header := labelFileHeader{}
	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	labels := make([]int8, header.NumLabels)

	for i := int32(0); i < header.NumLabels; i++ {
		err = binary.Read(reader, binary.BigEndian, &labels[i])
		if err != nil {
			return nil, err
		}
	}

	return labels, nil
}

func LoadSet(imageName, labelName string) (*Set, error) {
	images, err := LoadImages(imageName)

	if err != nil {
		return nil, err
	}

	labels, err := LoadLabels(labelName)

	if err != nil {
		return nil, err
	}

	if len(images) != len(labels) {
		return nil, ErrSize
	}

	return &Set{Images: images, Labels: labels}, nil
}

// At implements the image.Image interface.
func (img *Image) At(x, y int) color.Color {
	return color.Gray{Y: img[y*Width+x]}
}

// Set modifies the pixel at (x,y).
func (img *Image) Set(x, y int, v byte) {
	img[y*Width+x] = v
}

// Count returns the number of images and labels in the set.
func (s *Set) Count() int {
	return len(s.Labels)
}

// Get returns the i-th image and its label.
func (s *Set) Get(i int) (*Image, int8) {
	return s.Images[i], s.Labels[i]
}

func Load(dir string) (training, test *Set, err error) {
	training, err = LoadSet(path.Join(dir, TrainingImageFileName), path.Join(dir, TrainingLabelFileName))

	if err != nil {
		return nil, nil, err
	}

	test, err = LoadSet(path.Join(dir, TestImageFileName), path.Join(dir, TestLabelFileName))

	if err != nil {
		return nil, nil, err
	}

	return training, test, nil
}
