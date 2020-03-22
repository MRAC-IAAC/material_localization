// Last 0064

import java.io.File;
import java.util.Arrays;

PImage src;

// Well this is silly
String folder = "/../../../../data/drone_images/Camera/";

File[] filenames;

int imgId = 0;
int patchId = 0;

int windowSize = 100;

int category = 0;

String[] categoryNames = {"brick", "concrete", "metal", "wood", "z_none"};

int cursorX = -1;
int cursorY = -1;

public void setup() {
  size(1024, 678);

  folder = sketchPath() + folder;

  File directory = new File(folder);
  filenames = directory.listFiles();


  loadSrcImage();
  rectMode(CENTER);
}

public void draw() {
  image(src, 0, 0);

  int x = mouseX;
  int y = mouseY;
  if (x < windowSize / 2) {
    x = windowSize / 2;
  }
  if (y < windowSize / 2) {
    y = windowSize / 2;
  }
  if (x > width - (windowSize / 2)) {
    x = width - (windowSize / 2);
  }
  if (y > height - (windowSize / 2)) {
    y = height - (windowSize / 2);
  }

  cursorX = x;
  cursorY = y;

  stroke(255);
  noFill();
  rect(x, y, windowSize, windowSize);
}

public void keyPressed() {
  if (key==CODED) {
    if (keyCode == RIGHT) {
      imgId += 1;
      loadSrcImage();
    } else if (keyCode == LEFT) {
      imgId -= 1;
      loadSrcImage();
    }
  } else if (key == '1') {
    category = 0;
    patchId = 0;
    println(categoryNames[category]);
  } else if (key == '2') {
    patchId=  0;
    category = 1;
    println(categoryNames[category]);
  } else if (key == '3') {
    patchId = 0;
    category = 2;
    println(categoryNames[category]);
  } else if (key == '4') {
    patchId = 0;
    category = 3;
    println(categoryNames[category]);
  } else if (key == '5') {
    patchId = 0;
    category = 4;
    println(categoryNames[category]);
  } else {
    return;
  }
}

public void mousePressed() {
  String[] parts = filenames[imgId].toString().split("/");
  String name = parts[parts.length - 1];
  name = name.substring(0, name.length() - 4);
  name += "_" + patchId + ".png";
  name = categoryNames[category] + "/" + name;
  patchId  += 1;

  src.get(cursorX - (windowSize / 2), cursorY - (windowSize / 2), windowSize, windowSize).save("output/" + name);

  println(name);
}

public void loadSrcImage() {
  if (imgId < 0) {
    imgId = 0;
  }

  if (imgId >= filenames.length) {
    imgId = filenames.length - 1;
  }

  src = loadImage(filenames[imgId].toString());
  src.resize(1024, 0);
  println(filenames[imgId]);

  patchId = 0;
}