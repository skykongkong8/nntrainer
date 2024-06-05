// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_models.cpp
 * @date 25 Nov 2021
 * @brief unittest models for v2 version
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <ini_wrapper.h>
#include <memory>
#include <neuralnet.h>
#include <nntrainer_test_util.h>
#include <fc_layer_cl.h>

#include <models_golden_test.h>

using namespace nntrainer;

static inline constexpr const int NOT_USED_ = 1;

static IniSection nn_base("model", "type = NeuralNetwork");
static std::string fc_base = "type = Fully_connected";
static std::string red_mean_base = "type = reduce_mean";
static IniSection sgd_base("optimizer", "Type = sgd");
static IniSection constant_loss("loss", "type = constant_derivative");
static IniSection act_base("activation", "Type = Activation");

IniWrapper reduce_mean_last("reduce_mean_last",
                            {
                              nn_base + "batch_size=3",
                              sgd_base + "learning_rate=0.1",
                              IniSection("fc_1") + fc_base +
                                "unit=7 | input_shape=1:1:2",
                              IniSection("red_mean") + red_mean_base + "axis=3",
                              constant_loss,
                            });

IniWrapper fc_relu_decay(
  "fc_relu_decay",
  {nn_base + "Loss=mse | batch_size = 3", sgd_base + "learning_rate = 0.1",
   IniSection("input") + "type=input" + "input_shape = 1:1:3",
   IniSection("dense") + fc_base + "unit = 10" + "weight_decay=0.9",
   IniSection("act") + act_base + "Activation = relu",
   IniSection("dense_1") + fc_base + "unit = 2" + "bias_decay=0.9",
   IniSection("act_1") + act_base + "Activation = sigmoid"});

/**
 * @brief get function to make model with non-trainable fc layer
 * @param[in] idx index of the fc layer to be non-trainable
 * @retval function to make model with non-trainable fc layer
 */
std::function<std::unique_ptr<NeuralNetwork>()>
getFuncToMakeNonTrainableFc(int idx) {

  std::string fc1_trainable = (idx == 1) ? "trainable=false" : "trainable=false";
  std::string fc2_trainable = (idx == 2) ? "trainable=false" : "trainable=false";
  std::string fc3_trainable = (idx == 3) ? "trainable=false" : "trainable=false";

  return [fc1_trainable, fc2_trainable, fc3_trainable]() {
    std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());

    nn->setProperty({"batch_size=3"});

    auto outer_graph = makeGraph({
      {"input", {"name=in", "input_shape=1:1:3"}},
      {"fully_connected",
       {"name=fc1", "input_layers=in", "unit=10", "activation=relu",
        fc1_trainable}},
      {"fully_connected_cl",
       {"name=fc2", "input_layers=fc1", "unit=10", "activation=relu",
        fc2_trainable}},
      {"fully_connected_cl",
       {"name=fc3", "input_layers=fc2", "unit=2", "activation=sigmoid",
        fc3_trainable}},
      {"mse", {"name=loss", "input_layers=fc3"}},

    });

    // auto fc_cl_l = nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>({"name=fc1", "input_layers=in", "unit=10", "activation=relu",
    //     fc1_trainable});

    for (auto &node : outer_graph) {
      nn->addLayer(node);
    }

    // nn->setOptimizer(
    //   ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
    // nn->setProperty({"input_layers=in", "label_layers=loss"});

    return nn;
  };
}

static auto makeNonTrainableFcIdx1CL = getFuncToMakeNonTrainableFc(1);
static auto makeNonTrainableFcIdx2CL = getFuncToMakeNonTrainableFc(2);
// static auto makeNonTrainableFcIdx3 = getFuncToMakeNonTrainableFc(3);

GTEST_PARAMETER_TEST(
  model, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelIniTc(fc_relu_decay, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeNonTrainableFcIdx1CL, "non_trainable_fc_idx1",
                 ModelTestOption::NO_THROW_RUN),
    mkModelTc_V2(makeNonTrainableFcIdx2CL, "non_trainable_fc_idx2",
                 ModelTestOption::COMPARE_V2),
    // mkModelTc_V2(makeNonTrainableFcIdx2, "non_trainable_fc_idx2",
    //              ModelTestOption::ALL_V2),
    // mkModelTc_V2(makeNonTrainableFcIdx3, "non_trainable_fc_idx3",
    //              ModelTestOption::ALL_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info)
    -> const auto & { return std::get<1>(info.param); });

// TEST(nntrainerModelsCL, loadFromLayersBackbone_p) {
//   std::vector<std::shared_ptr<ml::train::Layer>> reference;
//   reference.emplace_back(
//     ml::train::layer::FullyConnectedCL({"name=fc1", "input_shape=3:1:2"}));
//   reference.emplace_back(
//     ml::train::layer::FullyConnectedCL({"name=fc2", "input_layers=fc1"}));
//   nntrainer::NeuralNetwork nn;
//   nn.addWithReferenceLayers(reference, "backbone", {}, {"fc1"}, {"fc2"},
//                             ml::train::ReferenceLayersType::BACKBONE, {});
//   nn.compile();
//   auto graph = nn.getFlatGraph();
//   for (unsigned int i = 0; i < graph.size(); ++i) {
//     EXPECT_EQ(graph.at(i)->getName(), "backbone/" + reference.at(i)->getName());
//   };
// }

#ifdef NDK_BUILD

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
#endif
