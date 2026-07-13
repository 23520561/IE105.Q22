import { useState } from "react";

const useModal = function (
  render: (closeHandler: VoidFunction) => React.ReactNode,
) {
  const [showModal, setShowModal] = useState(false);
  function customModal() {
    if (!showModal) {
      return null;
    }
    return render(() => toggleModal(false));
  }
  function toggleModal(state: boolean) {
    setShowModal(state);
  }
  return { customModal, toggleModal };
};
export default useModal;
