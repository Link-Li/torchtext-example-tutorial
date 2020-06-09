import torch
import sklearn as sk
import sklearn.metrics


def test_model(opt, emotion_model, test_iterator, critertion, save_model_path=None, last_F1=None):
    with torch.no_grad():
        emotion_model.eval()
        test_loss = 0
        y_true_test = []
        y_pre_test = []
        total_labels_test = 0
        for i, data in enumerate(test_iterator):
            
            input_texts, seq_lengths = data.sentence
            output = emotion_model(input_texts, seq_lengths)

            test_loss += critertion(output, data.label)
            _, predicted = torch.max(output, 1)
            y_true_test.extend(predicted.cpu())
            y_pre_test.extend(data.label.cpu())
            total_labels_test += data.label.size(0)

        test_loss /= total_labels_test
        test_correct = sk.metrics.accuracy_score(y_true_test, y_pre_test)
        test_F1 = sk.metrics.f1_score(y_true_test, y_pre_test, average='macro')
        test_R = sk.metrics.recall_score(y_true_test, y_pre_test, average='macro')
        test_precision = sk.metrics.precision_score(y_true_test, y_pre_test, average='macro')

        save_content = 'Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, test_loss: %f' % \
                       (test_correct, test_precision, test_R, test_F1, test_loss.item())
        print(save_content)

        if last_F1 is not None:
            if last_F1 < test_F1:
                if test_F1 > 0.69:
                    # 对合适的模型进行保存
                    model_name = str(test_F1)
                    torch.save(emotion_model.state_dict(), save_model_path + '/' + model_name + '.pth')
                    save_content = '**F1高于上次 %2f%%, 本次为了 %2f, 已经存储模型为 %s' % \
                                   (last_F1, test_F1, save_model_path + '/' + model_name + '.pth')
                    print(save_content)
                    print()
                else:
                    save_content = '**F1高于上次 %2f%%, 本次为了 %2f%%, 但低于69%%, 不存储' % (last_F1, test_F1)
                    print(save_content)
                    print()
                return test_F1
            else:
                save_content = 'F1低于上次 %2f, 本次为了 %2f' % (last_F1, test_F1)
                print(save_content)
                print()
            return last_F1
