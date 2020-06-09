import torch
import sklearn as sk
import sklearn.metrics


def test_model(emotion_model, test_loader, critertion, save_model_path=None, last_F1=None):
    with torch.no_grad():
        emotion_model.eval()
        run_loss = 0
        y_true = []
        y_pre = []
        total_labels = 0
        for i, data in enumerate(test_loader):
            output = emotion_model(data.sentence)

            run_loss += critertion(output, data.label)
            _, predicted = torch.max(output, 1)
            y_pre.extend(predicted.cpu())
            y_true.extend(data.label.cpu())
            total_labels += data.label.size(0)

        run_loss /= total_labels
        test_correct = sk.metrics.accuracy_score(y_true, y_pre)
        test_F1 = sk.metrics.f1_score(y_true, y_pre, average='macro')
        test_R = sk.metrics.recall_score(y_true, y_pre, average='macro')
        test_precision = sk.metrics.precision_score(y_true, y_pre, average='macro')

        save_content = 'Correct: %.5f, Precision: %.5f, R: %.5f, F1(macro): %.5f, test_loss: %f' % \
                       (test_correct, test_precision, test_R, test_F1, run_loss.item())
        print(save_content)

        if last_F1 is not None:
            if last_F1 < test_F1:
                if test_F1 > 0.71:
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
